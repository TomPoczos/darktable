/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
DOCUMENTATION
This module implements a scene-referred local contrast enhancement algorithm,
designed to enhance local details while preserving edges and avoiding artifacts.

It builds upon the original proof-of-concept algorithm proposed by WileCoyote:
https://discuss.pixls.us/t/experiments-with-a-scene-referred-local-contrast-module-proof-of-concept/55402

And then further explored and optimized by Christian Bouhon
https://discuss.pixls.us/t/contrast-management-rgb-a-new-scene-referred-approach-poc/56004

Current status as implemented by Jandren:
- Local contrast in log space based on the eigf surface blur filter.
*/

/* Work around a compiler bug, see
   https://github.com/darktable-org/darktable/issues/21801
*/
#if __GNUC__ == 16
#pragma GCC optimize ("no-ipa-cp")
#endif

#include "common/extra_optimizations.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/fast_guided_filter.h"
#include "common/color_picker.h"
#include "common/eigf.h"
#include "common/gaussian.h"
#include "common/luminance_mask.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/preview_data.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "dtgtk/paint.h"
#include "dtgtk/togglebutton.h"
#include "dtgtk/expander.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/choleski.h"
#include "iop/iop_api.h"
#include "libs/lib.h"
#include "common/iop_group.h"

#ifdef _OPENMP
#include <omp.h>
#endif

DT_MODULE_INTROSPECTION(2, dt_iop_contrast_params_t)

#define CT_BANDS 9          // detail levels 2..10, one node per octave
#define CT_BAND_D0 2.0f     // detail level of the coarsest node

// §1.9 FAST: how many of the *finest* (d->sigma[]-space, index 0) bands stay
// direct-eigf regardless of decomposition. phase0-hybrid-pyramid-summary.md's
// sweep labels this count `c` -- `H(5)` is the setting that measured safe on
// all six test images (5 finest direct, coarsest 9-5=4 pyramided when every
// band survives); implementation-plan.md §1.9's prose describes the same
// point as "the finest ~5 bands" but its formula ("cutover = nbands - 5")
// only equals 5 at nbands = 10, not this module's nbands = 9, so it is read
// here as a slip and this constant follows the tested number instead.
#define CT_FAST_DIRECT_BANDS 5
#define CT_FAST_MIN_DIM 8   // floor on a pyramid level's shorter side, px

// the graph: nodes run coarse (left) to fine (right), one per octave, so the
// x axis is simply k/CT_BANDS; the y axis is gain, soft-ranged to
// CT_GRAPH_Y_MAX to match the sliders' own soft range (gui_init) even though
// the hard range (band[]'s $MAX) reaches higher -- a node dragged past the
// top just rides the edge, exactly like the slider it drives.
#define CT_GRAPH_Y_MAX 2.0f
#define CT_GRAPH_RES 64      // curve points sampled between nodes, per implementation-plan.md §1.4

// the detail-scale ladder (picked-region measurement below, and the §2.1
// frame-wide one), shared by both: DoG bands per octave, the finest sigma,
// and how many octaves/bands the ladder is allowed to grow to.
#define CT_SCALES_PER_OCTAVE 3
#define CT_SIGMA_BASE 1.2f
#define CT_MAX_OCTAVES 12
#define CT_MAX_BANDS (CT_MAX_OCTAVES * CT_SCALES_PER_OCTAVE)

// §2.1: the frame-wide DoG ladder's block energy tables. Declared here,
// ahead of its own section further down, because dt_iop_contrast_gui_data_t
// needs the type; see that section for what builds and frees one.
typedef struct _ct_ladder_t
{
  int    nrungs;
  double lambda[CT_MAX_BANDS];  // band-centre wavelength (2*pi*sigma), level-0 pixels
  double step[CT_MAX_BANDS];    // level-0 pixels per pixel of the rung's own level
  size_t bw, bh;                 // block grid, the same for every rung
  double *sat2;                  // Sum(b^2) over blocks, nrungs * (bw+1) * (bh+1) doubles
  double *sat1;                  // Sum(|b|), same layout
  double noise_floor[CT_MAX_BANDS];  // §2.4: per-rung, frame-wide block-minimum noise estimate
} _ct_ladder_t;

typedef enum dt_iop_contrast_decomposition_t
{
  CT_DECOMPOSITION_ACCURATE = 0, // $DESCRIPTION: "accurate" -- every band direct, no pyramid
  CT_DECOMPOSITION_FAST = 1      // $DESCRIPTION: "fast" -- hybrid: coarse bands pyramided
} dt_iop_contrast_decomposition_t;

typedef struct dt_iop_contrast_params_t
{
  float gain_local_contrast;  // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0  $DESCRIPTION: "local contrast"
  float band[CT_BANDS];       // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0
  float scale_shift;          // $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0 $DESCRIPTION: "node placement"
  float edge_protection;      // $MIN: -10.0 $MAX: 10.0 $DEFAULT: 0.0 $DESCRIPTION: "adjust edge protection"
  int filter_iterations;      // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter iterations"
  float noise_bias;           // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.001 $DESCRIPTION: "noise bias"
  dt_iop_contrast_decomposition_t decomposition; // $DEFAULT: CT_DECOMPOSITION_ACCURATE $DESCRIPTION: "decomposition"
} dt_iop_contrast_params_t;

typedef struct dt_iop_contrast_data_t
{
  float gain_local_contrast;
  float scale_shift;
  float band[CT_BANDS];        // gains, copied verbatim from params
  int   nbands;                 // bands that survive the current roi scale
  float sigma[CT_BANDS];        // boundary sigmas, finest-surviving-first, in pixels of this roi
  float gain[CT_BANDS];         // matching gains, same order as sigma
  float feathering;
  int iterations;
  float noise_bias;
  dt_iop_contrast_decomposition_t decomposition;
} dt_iop_contrast_data_t;

// values 0 .. CT_BANDS-1 select one band's own raw b_k (param-space index,
// coarsest = 0); the two named values above that select an accumulated view
// instead (implementation-plan.md §1.6).
typedef enum dt_iop_details_display_t
{
  DT_CT_MASK_OFF = -1,
  DT_CT_MASK_CORRECTION = CT_BANDS,      // sum (g_k - 1) b_k, i.e. what the module is doing
  DT_CT_MASK_DETAIL     = CT_BANDS + 1   // sum b_k, i.e. the v1 behaviour
} dt_iop_details_display_t;

typedef struct dt_iop_contrast_gui_data_t
{
  // Flags
  dt_iop_details_display_t details_display;

  // GTK widgets
  GtkWidget *gain_local_contrast;
  GtkWidget *band[CT_BANDS];
  GtkWidget *scale_shift;
  GtkWidget *decomposition;
  GtkWidget *edge_protection;
  GtkWidget *filter_iterations;
  GtkWidget *noise_bias;

  // the graph (implementation-plan.md §1.4): a drawing area showing the nine
  // nodes as a curve, and a GtkStack toggled by middle-click on the graph
  // between that graph and the plain slider list.
  GtkDrawingArea *area;
  GtkStack *stack;
  dt_draw_curve_t *curve;

  // which bands survive at the current pipe scale, published from process()
  // under dt_iop_gui_enter/leave_critical_section (§1.5). nbands defaults to
  // CT_BANDS -- everything resolvable -- until the first preview pass lands.
  int nbands;
  float sigma[CT_BANDS];  // matching pixel sigma, index 0 = finest surviving

  // rms-based divisor of the currently displayed texture mask (a band or
  // DETAIL, never CORRECTION), published from process() the same way (§1.6)
  float mask_divisor;

  // graph interaction state
  gboolean dragging;
  int drag_band;   // node being dragged, -1 if none
  int hover_band;  // node nearest the pointer, -1 if none or not hovering

  // §2.2: the frame-wide DoG ladder's block SAT tables, republished through
  // dt_preview_data_t each untiled preview pass while the module is
  // expanded (§2.1's _build_ladder does the actual building). pd's buffer
  // is laid out node-major: (bw+1) x (bh+1) SAT nodes, 2*nrungs floats per
  // node (Sum(b^2), Sum(|b|) interleaved per rung) -- pd.width/height are
  // therefore the SAT dimensions, one more than the block grid on each
  // axis. ladder_nrungs/lambda/step are the ladder metadata dt_preview_data_t
  // has no room for; protected by the same self->gui_lock dt_preview_data_t
  // itself uses (dt_iop_gui_enter/leave_critical_section), since pd.module
  // == self.
  dt_preview_data_t pd;
  int ladder_nrungs;
  double ladder_lambda[CT_MAX_BANDS];
  double ladder_step[CT_MAX_BANDS];
  double ladder_noise_floor[CT_MAX_BANDS];  // §2.4, frame-wide, published the same way
  dt_iop_roi_t ladder_roi_in;  // the roi_in the ladder above was built from
} dt_iop_contrast_gui_data_t;


const char *name()
{
  return _("advanced contrast");
}

const char *aliases()
{
  return _("local contrast|texture|clarity|detail enhancement");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self, _("enhance local contrast by boosting details while preserving edges"),
     _("creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_EFFECTS;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  typedef struct dt_iop_contrast_params_v1_t
  {
    float gain_local_contrast;
    float detail_level;
    float edge_protection;
    int filter_iterations;
    float noise_bias;
  } dt_iop_contrast_params_v1_t;

  if(old_version == 1)
  {
    const dt_iop_contrast_params_v1_t *o = (dt_iop_contrast_params_v1_t *)old_params;
    dt_iop_contrast_params_t *n = malloc(sizeof(dt_iop_contrast_params_t));

    // v1 had one gain over everything finer than detail_level.
    // Reproduce it: put the ladder's coarsest boundary at detail_level and open
    // every band. The bands telescope, so this is exact up to the ladder's
    // half-octave quantisation, which scale_shift absorbs. The new master gain
    // stays neutral -- the old strength lives entirely in the opened bands now.
    const float d = CLAMP(o->detail_level, CT_BAND_D0, CT_BAND_D0 + CT_BANDS - 1);
    n->gain_local_contrast = 1.0f;
    n->scale_shift = CLAMP(d - roundf(d), -0.5f, 0.5f);
    for(int k = 0; k < CT_BANDS; k++)
      n->band[k] = (CT_BAND_D0 + k >= roundf(d)) ? o->gain_local_contrast : 1.0f;
    n->edge_protection = o->edge_protection;
    n->filter_iterations = o->filter_iterations;
    n->noise_bias = o->noise_bias;
    n->decomposition = CT_DECOMPOSITION_ACCURATE;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_contrast_params_t);
    *new_version = 2;
    return 0;
  }
  return 1;
}

// ---------------------------------------------------------------------------
// measuring the detail level from the ladder: the model and the fit
// ---------------------------------------------------------------------------
//
// §2.2: the ladder itself is built frame-wide now (§2.1's _build_ladder,
// queried through the SAT tables color_picker_apply() reads directly) rather
// than inside the picked box, so the box no longer bounds how far the ladder
// can reach -- what is left here is only the fit: given a rung's (wavelength,
// energy, weight) triples for the picked box, what texture size explains
// them. §2.3's `_fit_spectrum`, below, is that fit.
//
// detail_level is a low-pass size: the filter radius is 2^-detail_level of the
// image's long edge, and everything finer than that window ends up in the
// high-pass term this module boosts. so "the right detail level" is really the
// question "how large is the texture in this part of the picture?", and that
// is the classic scale-selection problem.
//
// Lindeberg's answer is to look for the scale at which a normalized derivative
// operator responds most strongly. for blob-like structure the operator is the
// scale-normalized Laplacian t^gamma * grad^2 L_t (t = sigma^2), and the
// normalization is what makes responses at different scales comparable at all:
// an unnormalized Laplacian simply decays with scale on any real image, so its
// argmax carries no information.
//
// gamma = 1 is the right exponent here, and not by convention. for an image
// whose power spectrum goes as f^-beta the expected energy of t*grad^2 L_t
// works out as t^(2 - (6-beta)/2) = t^((beta-2)/2): dead flat exactly when
// beta = 2, which is the spectrum natural images actually have. so on a patch
// of featureless fractal texture this measure has no opinion, and any peak it
// does report is a real departure from 1/f^2 -- a texture with a size --
// rather than an artifact of the falling spectrum.
//
// two conveniences make this cheap:
//
//  - a difference of Gaussians approximates the normalized Laplacian:
//    L_(k*sigma) - L_sigma ~ (k-1) * t * grad^2 L_t. k is constant along a
//    geometric ladder, so the mean squared DoG *is* the normalized Laplacian
//    energy up to one constant factor, and no explicit t^gamma appears.
//  - t*grad^2 is invariant under rescaling of the coordinates, so running the
//    same fixed sigma ladder on each level of a Gaussian pyramid produces
//    energies that are directly comparable across octaves -- and costs 4/3 of
//    a single full-resolution pass in total.
//
// the whole thing runs on log2 luminance, which is the space the module's own
// detail extraction works in, so the measurement sees exactly the signal that
// will be boosted -- including the shadow-noise suppression from the noise
// bias slider, since it is measured on the same biased luminance buffer.

// CT_SCALES_PER_OCTAVE / CT_SIGMA_BASE / CT_MAX_OCTAVES / CT_MAX_BANDS are
// declared near dt_iop_contrast_gui_data_t, above -- shared with §2.1's ladder.
// below this much detail there is nothing in the area worth enhancing and so
// nothing to size: 0.008 EV rms is half a percent of luminance, and even at
// this module's largest gain it stays under three percent, which is invisible.
// clear sky and out-of-focus background sit well below it; the grain of dry
// asphalt, about the faintest thing anyone would pick deliberately, sits
// comfortably above.
#define CT_FLAT_RMS_EV 0.008
#define CT_FLAT_ENERGY (CT_FLAT_RMS_EV * CT_FLAT_RMS_EV)
// grid resolution of the model fit's texture size, per octave
#define CT_FIT_STEPS_PER_OCTAVE 8.0
// the fit's other grid: the self-similar spectrum's slope, research.md §5.3's
// measured spread across scene categories
#define CT_FIT_BETA_MIN 1.6
#define CT_FIT_BETA_MAX 3.0
#define CT_FIT_BETA_STEPS 6  // 7 values, CT_FIT_BETA_MIN .. CT_FIT_BETA_MAX
// rungs this far below the peak are noise, and in log space they would
// otherwise dominate the residual
#define CT_ENERGY_FLOOR 1e-6
// how far an idealized hump is expected to sit from a real texture's ladder,
// as a fraction. it is the floor under every rung's error bar: past a couple
// of hundred independent samples a rung stops getting more trustworthy, so
// extra pixels stop buying it extra weight.
#define CT_MODEL_ERROR 0.15
// what the fit needs to be worth trusting, and so what decides the smallest
// area that can be measured at all. the hump is located by its flanks, so what
// matters is the *range* of sizes the ladder covers rather than how many rungs
// it took to cover them: one full octave, plus enough rungs to outnumber the
// model's three parameters. together these put the floor at 24 pixels on the
// short axis.
#define CT_MIN_SPAN 2.0
#define CT_MIN_BANDS 4

// §2.3: the exact DoG response (research.md §5.4c) of white noise smoothed
// to sigma_t = sqrt(tau), measured at scale s = sigma^2. reduces to the pure
// sensor-noise response (§5.4a) at tau = 0 -- which is what lets the same
// closed form serve as both the model's noise term and its texture term
// below, differing only in what tau each one carries.
static inline double _dog_shape(const double s, const double tau)
{
  const double k2 = 1.5874010519681994;  // 2^(2/3), the DoG step at CT_SCALES_PER_OCTAVE = 3
  return 1.0 / (2.0 * (k2 * s + tau))
       - 2.0 / ((k2 + 1.0) * s + 2.0 * tau)
       + 1.0 / (2.0 * (s + tau));
}

// research.md §5.4's model: E(s) = N*_dog_shape(s,0) + C*s^((beta-2)/2) +
// A*_dog_shape(s,tau) -- sensor noise, self-similar scene content, and a
// texture with a size, respectively. replaces `_fit_texture_scale`'s
// single-hump-plus-floor fit: that fit could locate a texture's size but had
// no way to tell a real one from a patch of self-similar content that simply
// disagreed with beta = 2, nor from a noise floor rising into the fine end
// of the ladder -- both read as "texture" before. N, C, A are amplitudes and
// so constrained >= 0; tau (texture size^2) and beta (spectral slope) are
// fit by grid search, same discipline as the argmax-is-hopeless reasoning
// `_fit_texture_scale` documented: locate the hump by fitting its whole
// shape in log energy, not by reading off a raw peak.
typedef struct _ct_fit_t
{
  double noise, self_similar, texture;  // amplitudes N, C, A -- all >= 0
  double tau, beta;                     // texture size^2, spectral slope
  double texture_peak;                  // texture*max(dog_shape) over the *measured* rungs, in energy units
  double residual;                      // weighted log-space residual, winning grid point
} _ct_fit_t;

// non-negative least squares for the model's (up to) three linear
// amplitudes, from the already-weighted normal equations AtA x = Aty.
// init_active marks which of the three are free to fit (§2.3's caller fixes
// the noise term off when it has a prior for it); n <= 3, so a closed-form
// Cramer's-rule solve per active set is simpler and cheaper than a generic
// solver at the thousands of tiny solves the grid search below runs -- drop
// the most negative component and resolve until every free component is
// non-negative or none are left.
static void _nnls3(const double AtA[3][3],
                   const double Aty[3],
                   const gboolean *const restrict init_active,
                   double *const restrict x)
{
  // whiten each column by its own RMS magnitude (sqrt of its AtA diagonal)
  // before solving. the three model terms live at wildly different natural
  // scales -- _dog_shape peaks around 1e-4, s^((beta-2)/2) spans several
  // orders of magnitude over a ten-octave ladder -- so a ridge or
  // determinant tolerance sized to matter for one column either does
  // nothing for the others or swamps them outright. normalized, every
  // active diagonal is ~1 and a single small, scale-free ridge and
  // determinant floor are meaningful for all three at once; the solved
  // coefficients are scaled back at the end.
  double norm[3];
  for(int a = 0; a < 3; a++) norm[a] = sqrt(fmax(AtA[a][a], 1e-300));

  double M[3][3], y[3];
  for(int a = 0; a < 3; a++)
  {
    y[a] = Aty[a] / norm[a];
    for(int b = 0; b < 3; b++) M[a][b] = AtA[a][b] / (norm[a] * norm[b]);
  }

  const double ridge = 1e-9;

  gboolean active[3] = { init_active[0], init_active[1], init_active[2] };

  for(int pass = 0; pass < 4; pass++)
  {
    int idx[3], k = 0;
    for(int j = 0; j < 3; j++) { x[j] = 0.0; if(active[j]) idx[k++] = j; }
    if(k == 0) return;

    double sol[3] = { 0.0, 0.0, 0.0 };
    if(k == 1)
    {
      const int a = idx[0];
      const double m = M[a][a] + ridge;
      if(m > 1e-12) sol[0] = y[a] / m;
    }
    else if(k == 2)
    {
      const int a = idx[0], b = idx[1];
      const double m00 = M[a][a] + ridge, m01 = M[a][b], m11 = M[b][b] + ridge;
      const double det = m00 * m11 - m01 * m01;
      if(det > 1e-12)
      {
        sol[0] = (y[a] * m11 - y[b] * m01) / det;
        sol[1] = (m00 * y[b] - m01 * y[a]) / det;
      }
    }
    else  // k == 3
    {
      const double m00 = M[0][0] + ridge, m01 = M[0][1], m02 = M[0][2];
      const double m11 = M[1][1] + ridge, m12 = M[1][2], m22 = M[2][2] + ridge;
      const double c00 = m11 * m22 - m12 * m12;
      const double c01 = m01 * m22 - m12 * m02;
      const double c02 = m01 * m12 - m11 * m02;
      const double det = m00 * c00 - m01 * c01 + m02 * c02;
      if(det > 1e-15)
      {
        const double y0 = y[0], y1 = y[1], y2 = y[2];
        sol[0] = (y0 * c00 - m01 * (y1 * m22 - m12 * y2) + m02 * (y1 * m12 - m11 * y2)) / det;
        sol[1] = (m00 * (y1 * m22 - m12 * y2) - y0 * c01 + m02 * (m01 * y2 - y1 * m02)) / det;
        sol[2] = (m00 * (m11 * y2 - y1 * m12) - m01 * (m01 * y2 - y1 * m02) + y0 * c02) / det;
      }
    }

    int worst = -1;
    double worst_val = -1e-9;  // small negative tolerance against float noise, in the whitened scale
    for(int c = 0; c < k; c++)
      if(sol[c] < worst_val) { worst_val = sol[c]; worst = idx[c]; }

    if(worst < 0)
    {
      for(int c = 0; c < k; c++) x[idx[c]] = fmax(sol[c], 0.0) / norm[idx[c]];
      return;
    }
    active[worst] = FALSE;
  }
}

// fit the model to one box's per-rung (wavelength, energy, weight) triples.
// noise_prior >= 0 fixes N to that value instead of fitting it (research.md
// §5.5: "prefer fixing N from the block-minimum noise estimate... stabilises
// everything else") -- §2.4 is what will supply a real prior; until then
// every caller passes -1 and N fits freely alongside C and A.
//
// returns FALSE under the same refusals `_fit_texture_scale` used: too few
// rungs or too narrow a span to trust a fit, or nothing above the noise
// floor anywhere in the box.
static gboolean _fit_spectrum(const double *const restrict lambda,
                              const double *const restrict energy,
                              const double *const restrict weight,
                              const int n,
                              const double noise_prior,
                              _ct_fit_t *const restrict fit)
{
  // the smallest area that can be measured at all covers exactly one octave,
  // so this comparison is met on the nose there and is given a rounding's
  // worth of slack rather than being left to turn on an ulp.
  if(n < CT_MIN_BANDS || lambda[n - 1] < CT_MIN_SPAN * lambda[0] * (1.0 - 1e-9))
    return FALSE;

  double peak_e = 0.0;
  for(int i = 0; i < n; i++) peak_e = fmax(peak_e, energy[i]);
  // nothing there at any scale: a blank sky, a blown highlight, a black frame
  if(peak_e <= CT_FLAT_ENERGY) return FALSE;

  double s[CT_MAX_BANDS];  // s = sigma^2, the model's own scale variable
  for(int i = 0; i < n; i++)
  {
    const double sigma_i = lambda[i] / (2.0 * M_PI);
    s[i] = sigma_i * sigma_i;
  }

  const gboolean fix_noise = noise_prior >= 0.0;
  const gboolean init_active[3] = { !fix_noise, TRUE, TRUE };

  // candidate texture sizes: from half the finest rung to the coarsest one,
  // same range `_fit_texture_scale` scanned and for the same reason -- the
  // top of it puts the hump's peak just past the end of the ladder, as far
  // as the rising flank alone can honestly be pushed.
  const double lo = (lambda[0] / (2.0 * M_PI)) * 0.5;
  const double hi = lambda[n - 1] / (2.0 * M_PI);
  const int tau_steps = MAX((int)(CT_FIT_STEPS_PER_OCTAVE * log2(hi / lo)), 1);

  double best_residual = DBL_MAX;
  _ct_fit_t best = { 0 };

  for(int bi = 0; bi <= CT_FIT_BETA_STEPS; bi++)
  {
    const double beta =
      CT_FIT_BETA_MIN + (double)bi * (CT_FIT_BETA_MAX - CT_FIT_BETA_MIN) / (double)CT_FIT_BETA_STEPS;

    double col_c[CT_MAX_BANDS];  // self-similar column depends only on beta
    for(int i = 0; i < n; i++) col_c[i] = pow(s[i], (beta - 2.0) * 0.5);

    for(int q = 0; q <= tau_steps; q++)
    {
      const double sigma_t = lo * exp2((double)q / CT_FIT_STEPS_PER_OCTAVE);
      const double tau = sigma_t * sigma_t;

      double col_n[CT_MAX_BANDS], col_a[CT_MAX_BANDS];
      for(int i = 0; i < n; i++)
      {
        col_n[i] = _dog_shape(s[i], 0.0);
        col_a[i] = _dog_shape(s[i], tau);
      }

      // 2-3 IRLS passes to approximate a log-space fit (research.md §5.5)
      // while keeping every inner solve linear: reweight by 1/E_model^2
      // after each solve, starting from the sampling weights alone.
      double w[CT_MAX_BANDS];
      for(int i = 0; i < n; i++) w[i] = weight[i];

      double x[3] = { 0.0, 0.0, 0.0 };
      for(int irls = 0; irls < 3; irls++)
      {
        double AtA[3][3] = { { 0.0 } };
        double Aty[3] = { 0.0, 0.0, 0.0 };
        for(int i = 0; i < n; i++)
        {
          const double y = fix_noise ? energy[i] - noise_prior * col_n[i] : energy[i];
          const double c[3] = { col_n[i], col_c[i], col_a[i] };
          for(int a = 0; a < 3; a++)
          {
            Aty[a] += w[i] * c[a] * y;
            for(int b = a; b < 3; b++) AtA[a][b] += w[i] * c[a] * c[b];
          }
        }
        AtA[1][0] = AtA[0][1]; AtA[2][0] = AtA[0][2]; AtA[2][1] = AtA[1][2];

        _nnls3(AtA, Aty, init_active, x);

        for(int i = 0; i < n; i++)
        {
          const double e_model =
            (fix_noise ? noise_prior : x[0]) * col_n[i] + x[1] * col_c[i] + x[2] * col_a[i];
          w[i] = weight[i] / fmax(e_model * e_model, CT_ENERGY_FLOOR * CT_ENERGY_FLOOR);
        }
      }

      double residual = 0.0;
      for(int i = 0; i < n; i++)
      {
        const double e_model =
          (fix_noise ? noise_prior : x[0]) * col_n[i] + x[1] * col_c[i] + x[2] * col_a[i];
        const double d = log(fmax(energy[i], peak_e * CT_ENERGY_FLOOR))
                        - log(fmax(e_model, peak_e * CT_ENERGY_FLOOR));
        residual += weight[i] * d * d;
      }

      if(residual < best_residual)
      {
        // A alone is not comparable to N or C: _dog_shape peaks around 1e-4
        // while the self-similar column can be O(1)-O(10), so a "large" A is
        // routinely needed just to explain a small amount of real energy --
        // and, at large tau, _dog_shape's near-zero, nearly featureless
        // values over every *measured* rung make the (tau, A) pair almost
        // unidentifiable from self-similar-only data: residual stays flat
        // while A drifts arbitrarily high chasing float-noise-scale
        // "improvement". texture_peak reports what A actually delivers over
        // the rungs this box could measure, in the same energy units
        // peak_e is in, which is what the caller below can honestly compare
        // against.
        double col_a_peak = 0.0;
        for(int i = 0; i < n; i++) col_a_peak = fmax(col_a_peak, col_a[i]);

        best_residual = residual;
        best.noise = fix_noise ? noise_prior : x[0];
        best.self_similar = x[1];
        best.texture = x[2];
        best.tau = tau;
        best.beta = beta;
        best.texture_peak = x[2] * col_a_peak;
        best.residual = residual;
      }
    }
  }

  if(best_residual == DBL_MAX) return FALSE;
  *fit = best;
  return TRUE;
}

// ---------------------------------------------------------------------------
// §2.1: the frame-wide DoG ladder + block energy tables
// ---------------------------------------------------------------------------
//
// research.md §5.1: build the measurement ladder over the whole frame instead
// of inside the picked box, and answer a box query against it in O(1) instead
// of rebuilding the ladder per pick. A band's value at a pixel already
// carries context from a sigma-neighbourhood, so a small box can still report
// on a large-wavelength band -- noisily, not truncated -- which is why the
// per-rung weights the fit uses (§2.3's `_fit_spectrum`, above) matter more
// here than they did before.
//
// `dt_gaussian_blur` (common/gaussian.c) is an IIR (van Vliet) approximation
// whose cost is independent of sigma, unlike the explicit mirrored kernels
// the picked-region ladder used to need to bound its own reach. That removes
// the old ladder-length ceiling outright; what still bounds this ladder is
// simply running out of pixels to decimate into, not the cost of a wide blur.
//
// §2.2 republishes the ladder this builds through dt_preview_data_t and
// wires color_picker_apply() to query it synchronously -- see that section,
// below, for the box query and the fit it feeds (§2.3's `_fit_spectrum`,
// above).

#define CT_BLOCK 8            // block-statistics granularity, in level-0 (finest rung) pixels
#define CT_LADDER_MIN_DIM 4   // stop decimating once the next level would be smaller than this

static void _ladder_free(_ct_ladder_t *const ladder)
{
  dt_free_align(ladder->sat2);
  dt_free_align(ladder->sat1);
  memset(ladder, 0, sizeof(_ct_ladder_t));
}

// accumulate one rung's band into the ladder's fixed bw x bh block grid.
// block (bx, by) covers level-0 pixels [bx*CT_BLOCK, (bx+1)*CT_BLOCK) x (...),
// which on this rung's own (possibly decimated) level maps to
// [bx*CT_BLOCK/step, (bx+1)*CT_BLOCK/step) x (...) -- several level pixels
// wide before the ladder has decimated past CT_BLOCK, one or less afterwards.
// using one grid for every rung, rather than a grid sized to each rung's own
// resolution, is what lets a box query be 4 lookups regardless of which rung
// it is asking about (research.md §5.2): the caller never needs to know a
// rung's own resolution to query it.
static void _ladder_accumulate_blocks(const float *const restrict band,
                                      const size_t cw, const size_t ch,
                                      const double step,
                                      const size_t bw, const size_t bh,
                                      double *const restrict blk2,
                                      double *const restrict blk1)
{
  DT_OMP_FOR()
  for(size_t by = 0; by < bh; by++)
  {
    const size_t y0 = MIN((size_t)((double)(by * CT_BLOCK) / step), ch);
    size_t y1 = MIN((size_t)ceil((double)((by + 1) * CT_BLOCK) / step), ch);
    if(y1 <= y0) y1 = MIN(y0 + 1, ch);

    for(size_t bx = 0; bx < bw; bx++)
    {
      double s2 = 0.0, s1 = 0.0;
      if(y0 < ch)
      {
        const size_t x0 = MIN((size_t)((double)(bx * CT_BLOCK) / step), cw);
        size_t x1 = MIN((size_t)ceil((double)((bx + 1) * CT_BLOCK) / step), cw);
        if(x1 <= x0) x1 = MIN(x0 + 1, cw);

        if(x0 < cw)
          for(size_t j = y0; j < y1; j++)
          {
            const float *const row = band + j * cw;
            for(size_t i = x0; i < x1; i++)
            {
              const double v = (double)row[i];
              s2 += v * v;
              s1 += fabs(v);
            }
          }
      }
      blk2[by * bw + bx] = s2;
      blk1[by * bw + bx] = s1;
    }
  }
}

// standard summed-area table, one row/column of zero padding on the low side
// so a box query is sat[y1][x1] - sat[y0][x1] - sat[y1][x0] + sat[y0][x0]
// with no special-casing at the edges. kept in double precision -- the whole
// point of accumulating it once per ladder rather than per pick -- because a
// query can sum thousands of blocks and this is exactly the kind of
// running sum that drifts in float.
static void _ladder_build_sat(const double *const restrict blk,
                              const size_t bw, const size_t bh,
                              double *const restrict sat)
{
  const size_t sw = bw + 1;
  for(size_t x = 0; x <= bw; x++) sat[x] = 0.0;
  for(size_t y = 1; y <= bh; y++)
  {
    sat[y * sw] = 0.0;
    double rowsum = 0.0;
    for(size_t x = 1; x <= bw; x++)
    {
      rowsum += blk[(y - 1) * bw + (x - 1)];
      sat[y * sw + x] = sat[(y - 1) * sw + x] + rowsum;
    }
  }
}

// §2.4/research.md §5.2, §5.5: this rung's noise floor -- the minimum block
// energy among blocks whose L2/L1 ratio (kappa) is close to what pure
// Gaussian noise gives. Restricting to near-Gaussian blocks is what keeps a
// block that happens to hold real texture or a hard edge -- either one pushes
// kappa well away from CT_KAPPA_GAUSSIAN -- from dragging the floor down
// below the sensor's actual noise level. Computed frame-wide, over every
// block on the rung, not just a picked box: "noise is global, texture is
// local" (implementation-plan.md §2.4), so this is a far more stable
// estimate than anything one box could produce on its own.
#define CT_KAPPA_GAUSSIAN 1.2533141373155003  // sqrt(pi/2), kappa of Gaussian noise
#define CT_KAPPA_NOISE_TOL 0.10                // +/- 10% of CT_KAPPA_GAUSSIAN counts as "noise-like"
#define CT_KAPPA_EDGE 2.0                      // sparseness warning threshold, research.md §5.2/§5.9
#define CT_NOISE_DOMINATED_FRAC 0.15           // S/(S+N) below this at the box's peak rung -> warn

static double _ladder_rung_noise_floor(const double *const restrict blk2,
                                       const double *const restrict blk1,
                                       const size_t nblocks,
                                       const double step)
{
  const double n_per_block = fmax(1.0, (CT_BLOCK / step) * (CT_BLOCK / step));
  double floor_e = -1.0;
  for(size_t i = 0; i < nblocks; i++)
  {
    const double m = blk1[i] / n_per_block;
    if(m <= 0.0) continue;
    const double e = blk2[i] / n_per_block;
    const double kappa = sqrt(e) / m;
    if(fabs(kappa - CT_KAPPA_GAUSSIAN) <= CT_KAPPA_NOISE_TOL * CT_KAPPA_GAUSSIAN
       && (floor_e < 0.0 || e < floor_e))
      floor_e = e;
  }
  return floor_e;  // < 0: no near-Gaussian block found on this rung
}

// §2.4: fold every rung's own noise floor into a single scalar estimate of
// the fit's N (research.md §5.5's "block-minimum noise estimate"). Each
// rung's floor is a raw energy; dividing out that rung's own noise transfer
// function (_dog_shape(s,0)) recovers what N it implies, and the minimum
// implied N across the ladder is the least contaminated one -- self-similar
// or textured content only ever adds energy on top of the true noise floor,
// never subtracts from it.
static double _ladder_estimate_noise(const double *const restrict lambda,
                                     const double *const restrict noise_floor,
                                     const int nrungs)
{
  double best = -1.0;
  for(int r = 0; r < nrungs; r++)
  {
    if(noise_floor[r] < 0.0) continue;
    const double sigma = lambda[r] / (2.0 * M_PI);
    const double g0 = _dog_shape(sigma * sigma, 0.0);
    if(g0 <= 0.0) continue;
    const double n_est = noise_floor[r] / g0;
    if(best < 0.0 || n_est < best) best = n_est;
  }
  return best;  // < 0: no usable rung, caller's fit falls back to fitting N freely
}

// build the ladder frame-wide: mean-centre log2 luminance once, then climb
// octaves, each one's four rungs blurred straight from that octave's own
// base (never chained rung-to-rung, which is what keeps every rung an
// honest measurement of the base rather than of an already-smoothed
// approximation of it), decimating by two between octaves once the base is
// safely past its own Nyquist frequency. There is no small-area floor to
// fall back from here, unlike a picked-region ladder: a preview frame is
// always large enough to decimate, so this always does (research.md §5.1).
static gboolean _build_ladder(const float *const restrict lum,
                              const size_t width, const size_t height,
                              _ct_ladder_t *const ladder)
{
  memset(ladder, 0, sizeof(_ct_ladder_t));
  if(width < CT_BLOCK || height < CT_BLOCK) return FALSE;

  ladder->bw = (width + CT_BLOCK - 1) / CT_BLOCK;
  ladder->bh = (height + CT_BLOCK - 1) / CT_BLOCK;
  const size_t sat_stride = (ladder->bw + 1) * (ladder->bh + 1);
  const size_t npixels = width * height;

  double *const restrict sat2 = dt_alloc_align_double(sat_stride * CT_MAX_BANDS);
  double *const restrict sat1 = dt_alloc_align_double(sat_stride * CT_MAX_BANDS);
  double *const restrict blk2 = dt_alloc_align_double(ladder->bw * ladder->bh);
  double *const restrict blk1 = dt_alloc_align_double(ladder->bw * ladder->bh);
  float *restrict level = dt_alloc_align_float(npixels);   // this octave's base
  float *restrict next = dt_alloc_align_float(npixels);    // next octave's decimated base
  float *restrict band = dt_alloc_align_float(npixels);
  float *restrict rung[CT_SCALES_PER_OCTAVE + 1] = { 0 };

  gboolean ok = sat2 && sat1 && blk2 && blk1 && level && next && band;
  for(int s = 0; ok && s <= CT_SCALES_PER_OCTAVE; s++)
  {
    rung[s] = dt_alloc_align_float(npixels);
    ok = ok && rung[s];
  }

  if(!ok)
  {
    dt_free_align(sat2); dt_free_align(sat1);
    dt_free_align(blk2); dt_free_align(blk1);
    dt_free_align(level); dt_free_align(next); dt_free_align(band);
    for(int s = 0; s <= CT_SCALES_PER_OCTAVE; s++) dt_free_align(rung[s]);
    memset(ladder, 0, sizeof(_ct_ladder_t));
    return FALSE;
  }

  // mean-centre log2 luminance once, over the whole frame -- everything
  // downstream is a difference of blurs and so ignores this offset
  // mathematically, but centring it keeps the numbers well away from
  // wherever the exposure happens to put the absolute level (the same
  // exposure-invariance argument as `_fit_spectrum`'s log-energy fit,
  // above -- both want the ladder's numbers independent of exposure).
  double mean_ev = 0.0;
  DT_OMP_FOR(reduction(+ : mean_ev))
  for(size_t k = 0; k < npixels; k++)
  {
    level[k] = log2f(fmaxf(lum[k], NORM_MIN));
    mean_ev += level[k];
  }
  mean_ev /= (double)npixels;
  const float offset = (float)mean_ev;
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++) level[k] -= offset;

  size_t cw = width, ch = height;
  double step = 1.0;
  int nrungs = 0;

  for(int octave = 0;
      ok && octave < CT_MAX_OCTAVES && nrungs + CT_SCALES_PER_OCTAVE <= CT_MAX_BANDS;
      octave++)
  {
    const float minv = -1.0e6f, maxv = 1.0e6f;
    for(int s = 0; s <= CT_SCALES_PER_OCTAVE; s++)
    {
      const float sigma = CT_SIGMA_BASE * exp2f((float)s / CT_SCALES_PER_OCTAVE);
      dt_gaussian_t *const g =
        dt_gaussian_init((int)cw, (int)ch, 1, &maxv, &minv, sigma, DT_IOP_GAUSSIAN_ZERO);
      if(!g) { ok = FALSE; break; }
      dt_gaussian_blur(g, level, rung[s]);
      dt_gaussian_free(g);
    }
    if(!ok) break;

    for(int s = 0; s < CT_SCALES_PER_OCTAVE; s++)
    {
      DT_OMP_FOR()
      for(size_t k = 0; k < cw * ch; k++) band[k] = rung[s][k] - rung[s + 1][k];

      _ladder_accumulate_blocks(band, cw, ch, step, ladder->bw, ladder->bh, blk2, blk1);
      ladder->noise_floor[nrungs] =
        _ladder_rung_noise_floor(blk2, blk1, ladder->bw * ladder->bh, step);
      _ladder_build_sat(blk2, ladder->bw, ladder->bh, sat2 + (size_t)nrungs * sat_stride);
      _ladder_build_sat(blk1, ladder->bw, ladder->bh, sat1 + (size_t)nrungs * sat_stride);

      // the DoG's peak frequency sits within a percent of the geometric mean
      // of its two rung sigmas; 2*pi*sigma is the sigma-to-wavelength
      // convention `_fit_spectrum`'s model (tau, s = sigma^2) already established.
      const double sigma_s = CT_SIGMA_BASE * exp2((double)s / CT_SCALES_PER_OCTAVE);
      const double sigma_s1 = CT_SIGMA_BASE * exp2((double)(s + 1) / CT_SCALES_PER_OCTAVE);
      ladder->lambda[nrungs] = 2.0 * M_PI * sqrt(sigma_s * sigma_s1) * step;
      ladder->step[nrungs] = step;
      nrungs++;
    }

    const size_t nw = cw / 2, nh = ch / 2;
    if(nw < CT_LADDER_MIN_DIM || nh < CT_LADDER_MIN_DIM) break;

    // the coarsest rung of this octave (index CT_SCALES_PER_OCTAVE, sigma
    // twice the base) is already blurred well past the new grid's Nyquist
    // frequency, so plain subsampling is exact enough -- the same B'
    // principle phase0-nonrecursive-pyramid.md validated, applied here to a
    // pure measurement, not to anything the halo-risk finding was about.
    DT_OMP_FOR()
    for(size_t j = 0; j < nh; j++)
      for(size_t i = 0; i < nw; i++)
        next[j * nw + i] = rung[CT_SCALES_PER_OCTAVE][(2 * j) * cw + 2 * i];

    float *const swap = level; level = next; next = swap;
    cw = nw; ch = nh;
    step *= 2.0;
  }

  ladder->nrungs = nrungs;

  dt_free_align(blk2); dt_free_align(blk1);
  dt_free_align(level); dt_free_align(next); dt_free_align(band);
  for(int s = 0; s <= CT_SCALES_PER_OCTAVE; s++) dt_free_align(rung[s]);

  if(!ok || nrungs == 0)
  {
    dt_free_align(sat2); dt_free_align(sat1);
    memset(ladder, 0, sizeof(_ct_ladder_t));
    return FALSE;
  }

  ladder->sat2 = sat2;
  ladder->sat1 = sat1;
  return TRUE;
}

// §2.2: dt_preview_data_fill_t for publishing a just-built ladder. Reshapes
// the ladder's rung-major SAT tables (one (bw+1)x(bh+1) grid per rung) into
// the node-major, per-node-interleaved layout dt_preview_data_t expects
// (`components` floats per "pixel", here per SAT node): 2*nrungs floats per
// node, Sum(b^2)/Sum(|b|) for rung 0, then rung 1, and so on. A cheap
// reshape, not a rebuild -- the ladder itself was already built outside the
// GUI lock, which is what this fill runs under (dt_preview_data_store's
// contract: fill() must be cheap).
static void _ladder_fill_cb(void *const user_data, float *const buf, const size_t nelems)
{
  const _ct_ladder_t *const ladder = (const _ct_ladder_t *)user_data;
  const size_t sw = ladder->bw + 1, sh = ladder->bh + 1;
  const size_t comps = (size_t)(2 * ladder->nrungs);
  (void)nelems;  // == sw * sh * comps, by construction of the caller's resize

  for(size_t y = 0; y < sh; y++)
    for(size_t x = 0; x < sw; x++)
    {
      float *const dst = buf + (y * sw + x) * comps;
      for(int r = 0; r < ladder->nrungs; r++)
      {
        dst[2 * r]     = (float)ladder->sat2[(size_t)r * sw * sh + y * sw + x];
        dst[2 * r + 1] = (float)ladder->sat1[(size_t)r * sw * sh + y * sw + x];
      }
    }
}

// Compute pixel-wise luminance (no boost) and add the noise bias, exactly as
// the detail ladder below expects to see it.
__DT_CLONE_TARGETS__
static inline void compute_luminance(const float *const restrict in,
                                     float *const restrict luminance,
                                     const dt_iop_roi_t *const roi_in,
                                     const dt_iop_contrast_data_t *const d)
{
  const size_t width = (size_t)roi_in->width;
  const size_t height = (size_t)roi_in->height;
  const size_t npixels = width * height;

  luminance_mask(in, luminance, width, height, DT_TONEEQ_NORM_2, 1.0f, 0.0f, 1.0f);
  const float noise_bias = d->noise_bias;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    luminance[k] += noise_bias;
  }
}

// §1.9 FAST: the coarse tail of the ladder, from research.md §2.4 option B'
// (phase0-nonrecursive-pyramid.md's winning variant), restricted to the
// coarse bands per phase0-hybrid-pyramid.md's finding that the pyramid
// softens dense fine texture but is safe -- and where direct (A)'s own cost
// concentrates, since sigma is largest there -- on the coarse ones.
//
// Builds one shared chain of bases by repeatedly halving the untouched
// full-resolution luminance -- never a level's own blurred output, that
// recursive variant is what phase0-pyramid.md rejected in the first place --
// runs one eigf call per level at that level's own (shrunk) resolution, and
// upsamples the result back to full res via interpolate_bilinear() before
// folding it into the same accumulate the direct bands use.
//
// The ladder is an exact octave-per-band geometric progression
// (modify_roi_in, §1.2), so doubling the decimation once per band -- 2x for
// the first pyramid band, 4x for the next, and so on -- keeps every level's
// *local* sigma close to constant by construction, matching phase0-
// nonrecursive-pyramid.md's tested recipe (sigma_local ~= 2.5px at every
// level) without needing to compute a target and round to it.
//
// full_scratch is the caller's full-res "blur" buffer: already allocated,
// and the direct loop above is done with it by the time this runs.
__DT_CLONE_TARGETS__
static void _accumulate_pyramid_bands(const float *const restrict lum,
                                      const float *const restrict log_lum,
                                      float *const restrict correction,
                                      float *const restrict coarsest,
                                      float *const restrict full_scratch,
                                      const size_t width, const size_t height,
                                      const dt_iop_contrast_data_t *const d,
                                      const int direct_bands,
                                      const int display_band)
{
  const size_t npixels = width * height;

  float *restrict base = NULL;           // owned decimated base, replaced each level
  const float *restrict base_src = lum;  // this level's source to downsample from
  size_t base_w = width, base_h = height;
  int decimation = 1;

  for(int k = direct_bands; k < d->nbands; k++)
  {
    const size_t new_w = MAX(base_w / 2, (size_t)CT_FAST_MIN_DIM);
    const size_t new_h = MAX(base_h / 2, (size_t)CT_FAST_MIN_DIM);
    if(new_w < base_w && new_h < base_h)
    {
      float *const restrict next = dt_alloc_align_float(new_w * new_h);
      if(next)
      {
        interpolate_bilinear(base_src, base_w, base_h, next, new_w, new_h, 1);
        dt_free_align(base);
        base = next;
        base_src = base;
        base_w = new_w;
        base_h = new_h;
        decimation *= 2;
      }
    }
    // else: hit the size floor -- keep reusing this level for every
    // remaining (coarser) band. their local sigma runs a bit above target,
    // which is not a correctness problem, just slightly less separation
    // between them; only happens on very small previews/exports.

    float *const restrict level_blur = dt_alloc_align_float(base_w * base_h);
    if(!level_blur) break;

    memcpy(level_blur, base_src, base_w * base_h * sizeof(float));
    const float local_sigma = fmaxf(d->sigma[k] / (float)decimation, 1.0f);
    fast_eigf_surface_blur(level_blur, base_w, base_h, local_sigma, d->feathering, d->iterations,
                           DT_GF_BLENDING_LINEAR, 1.0f,
                           0.0f, NORM_MIN, 4.0f);
    interpolate_bilinear(level_blur, base_w, base_h, full_scratch, width, height, 1);
    dt_free_align(level_blur);

    const float gain_minus_one = d->gain[k] - 1.0f;
    const gboolean is_display = (display_band == k);
    const gboolean detail_mode = (display_band == -2);

    DT_OMP_FOR()
    for(size_t p = 0; p < npixels; p++)
    {
      const float b_k = log_lum[p] - log2f(fmaxf(full_scratch[p], NORM_MIN));
      if(is_display) correction[p] = b_k;
      else if(detail_mode) correction[p] += b_k;
      else if(display_band < 0) correction[p] += gain_minus_one * b_k;
    }

    if(k == d->nbands - 1) memcpy(coarsest, full_scratch, npixels * sizeof(float));
  }

  dt_free_align(base);
}

// the ladder: every band a direct, full-resolution eigf call against the
// untouched luminance, accumulated as it goes so no band is ever stored
// (research.md §2.4 option A; see phase0-hybrid-pyramid.md for why this is
// the only decomposition that ships as the accurate default).
//
// correction ends up holding sum_k (gain_k - 1) * b_k, in EV, still missing
// the master gain and the Wiener gate -- both are cheap scalar-per-pixel
// operations applied once by the caller, rather than folded in here.
//
// display_band selects what correction ends up holding (Phase 1.6):
// >= 0 writes that one surviving band's raw b_k, instead of accumulating,
// so the per-band mask view can reuse this same pass rather than a second
// traversal; -2 accumulates the unweighted sum of every band's b_k (the
// DETAIL view, i.e. v1's own behaviour, with no gain applied); -1 (or
// anything else negative) is the normal gain-weighted accumulate, which
// doubles as the CORRECTION view before the caller's gate and master gain.
//
// §1.9: when d->decomposition == FAST, only the finest CT_FAST_DIRECT_BANDS
// bands take the direct path above; d->nbands - CT_FAST_DIRECT_BANDS coarser
// bands are missing here, filled in below by _accumulate_pyramid_bands().
// ACCURATE always runs every band direct -- direct_bands == d->nbands, the
// loop is untouched from before this section existed, and the call below
// never executes, so ACCURATE stays bit-identical to 1.3.
__DT_CLONE_TARGETS__
static void _decompose_and_accumulate(const float *const restrict lum,
                                      float *const restrict correction,
                                      float *const restrict coarsest,
                                      const size_t width, const size_t height,
                                      const dt_iop_contrast_data_t *const d,
                                      const int display_band)
{
  const size_t npixels = width * height;

  memcpy(coarsest, lum, npixels * sizeof(float));
  memset(correction, 0, npixels * sizeof(float));

  float *const restrict log_lum = dt_alloc_align_float(npixels);
  float *const restrict blur = dt_alloc_align_float(npixels);
  if(!log_lum || !blur)
  {
    dt_free_align(log_lum);
    dt_free_align(blur);
    return;
  }

  DT_OMP_FOR()
  for(size_t p = 0; p < npixels; p++)
    log_lum[p] = log2f(fmaxf(lum[p], NORM_MIN));

  const int direct_bands
    = (d->decomposition == CT_DECOMPOSITION_FAST) ? MIN(CT_FAST_DIRECT_BANDS, d->nbands) : d->nbands;

  for(int k = 0; k < direct_bands; k++)
  {
    memcpy(blur, lum, npixels * sizeof(float));
    fast_eigf_surface_blur(blur, width, height, d->sigma[k], d->feathering, d->iterations,
                           DT_GF_BLENDING_LINEAR, 1.0f,
                           0.0f, NORM_MIN, 4.0f);

    const float gain_minus_one = d->gain[k] - 1.0f;
    const gboolean is_display = (display_band == k);
    const gboolean detail_mode = (display_band == -2);

    DT_OMP_FOR()
    for(size_t p = 0; p < npixels; p++)
    {
      const float b_k = log_lum[p] - log2f(fmaxf(blur[p], NORM_MIN));
      if(is_display) correction[p] = b_k;
      else if(detail_mode) correction[p] += b_k;
      else if(display_band < 0) correction[p] += gain_minus_one * b_k;
    }

    if(k == d->nbands - 1) memcpy(coarsest, blur, npixels * sizeof(float));
  }

  if(direct_bands < d->nbands)
    _accumulate_pyramid_bands(lum, log_lum, correction, coarsest, blur, width, height,
                              d, direct_bands, display_band);

  dt_free_align(log_lum);
  dt_free_align(blur);
}

// the Wiener gate, gauged off the coarsest band -- the closest thing this
// module has to v1's single smoothed reference -- and applied once to the
// whole accumulated correction rather than per band (research.md's noise
// model is about the local signal level, which the coarsest band already
// estimates about as well as any finer one would).
__DT_CLONE_TARGETS__
static inline float _wiener_gate(const float coarsest_pixel, const float noise_bias)
{
  const float noise_power = noise_bias * noise_bias;
  const float combined_power = coarsest_pixel * coarsest_pixel;
  return fmaxf(combined_power - noise_power, 0.0f) / fmaxf(combined_power, NORM_MIN);
}

// Apply the accumulated, gated correction in linear space.
__DT_CLONE_TARGETS__
static inline void apply_correction(const float *const restrict in,
                                    const float *const restrict correction_ev,
                                    float *const restrict out,
                                    const dt_iop_roi_t *const roi_in)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float multiplier = exp2f(correction_ev[k]);
    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
    out[4 * k + 3] = in[4 * k + 3];
  }
}

/*
 Display a mask -- either the correction (what the module is doing to each
 pixel) or a raw band/DETAIL texture view. Output is a grayscale image
 normalized to [0, 1] where 0.5 = no signal, < 0.5 = negative, > 0.5 =
 positive.

 divisor rescales correction_ev before the sigmoid: 1.0 for CORRECTION,
 where the values are already in the module's own EV units, or a
 frame-wide rms (research.md §4.2) for a raw band/DETAIL view, which
 without it reads as flat grey -- those are un-gained EV differences with
 no fixed scale of their own (implementation-plan.md §1.6).
 */
__DT_CLONE_TARGETS__
static inline void display_correction_mask(const float *const restrict correction_ev,
                                           const float divisor,
                                           float *const restrict out,
                                           const dt_iop_roi_t *const roi_in)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float y = correction_ev[k] / divisor;
    const float intensity = y / sqrtf(y * y + 1.0f) * 0.5f + 0.5f; // Smooth mapping to [0, 1]

    for_each_channel(c)
    {
      out[4 * k + c] = intensity;
    }
    out[4 * k + 3] = 1.0f;
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid,
             void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_contrast_data_t *const d = piece->data;
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  // publish which bands survive at this pipe's scale, for the graph's
  // stripe shading (§1.5). keyed on the FULL pipe, not the preview: the
  // preview runs at a fixed thumbnail scale that has nothing to do with the
  // zoom level the graph is meant to describe.
  if(g && self->dev->gui_attached && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL))
  {
    dt_iop_gui_enter_critical_section(self);
    g->nbands = d->nbands;
    memcpy(g->sigma, d->sigma, sizeof(g->sigma));
    dt_iop_gui_leave_critical_section(self);
  }

  const float *const restrict in = (float *const)ivoid;
  float *const restrict out = (float *const)ovoid;

  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const size_t npixels = width * height;

  float *restrict luminance = dt_alloc_align_float(npixels);
  float *restrict correction = dt_alloc_align_float(npixels);
  float *restrict coarsest = dt_alloc_align_float(npixels);

  if(!luminance || !correction || !coarsest)
  {
    dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
    dt_free_align(luminance);
    dt_free_align(correction);
    dt_free_align(coarsest);
    return;
  }

  compute_luminance(in, luminance, roi_in, d);

  // §2.2: keep the frame-wide DoG ladder + its published SAT tables current
  // while the module is expanded, on the untiled preview pipe -- a tile is
  // not a whole frame. color_picker_apply() queries this synchronously on
  // the GUI thread; see there for the box query and the fit it feeds.
  if(g && self->dev->gui_attached && self->expanded
     && dt_pipe_is_preview(piece->pipe) && !piece->pipe->tiling)
  {
    _ct_ladder_t built;
    if(_build_ladder(luminance, width, height, &built))
    {
      // components must be set before dt_preview_data_store()'s resize
      // check runs, and that check only looks at width/height -- so force a
      // resize (by invalidating the stored dimensions) whenever nrungs, and
      // so components, has changed since the last publish.
      dt_iop_gui_enter_critical_section(self);
      if(g->pd.components != (size_t)(2 * built.nrungs))
      {
        g->pd.components = (size_t)(2 * built.nrungs);
        g->pd.width = 0;
        g->pd.height = 0;
      }
      dt_iop_gui_leave_critical_section(self);

      dt_preview_data_store(&g->pd, built.bw + 1, built.bh + 1, piece, _ladder_fill_cb, &built);

      dt_iop_gui_enter_critical_section(self);
      g->ladder_nrungs = built.nrungs;
      memcpy(g->ladder_lambda, built.lambda, sizeof(g->ladder_lambda));
      memcpy(g->ladder_step, built.step, sizeof(g->ladder_step));
      memcpy(g->ladder_noise_floor, built.noise_floor, sizeof(g->ladder_noise_floor));
      g->ladder_roi_in = *roi_in;
      dt_iop_gui_leave_critical_section(self);
    }
    _ladder_free(&built);
  }

  // Phase 1.6: map the GUI's mask selection onto _decompose_and_accumulate's
  // display_band. CORRECTION and the normal (non-display) path are the same
  // -1 accumulate -- CORRECTION is exactly that sum before the gate/master
  // gain below, "what the module is doing". a band index that isn't
  // currently resolvable (>= d->nbands, greyed out in the graph) has no
  // slot in d->sigma/d->gain; fall back to CORRECTION rather than show
  // nothing.
  const gboolean showing_mask =
    g && g->details_display != DT_CT_MASK_OFF && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL);
  int display_band = -1;
  gboolean showing_texture = FALSE;
  if(showing_mask)
  {
    if(g->details_display == DT_CT_MASK_DETAIL) { display_band = -2; showing_texture = TRUE; }
    else if(g->details_display != DT_CT_MASK_CORRECTION && g->details_display < d->nbands)
    {
      display_band = d->nbands - 1 - g->details_display;
      showing_texture = TRUE;
    }
  }

  _decompose_and_accumulate(luminance, correction, coarsest, width, height, d, display_band);

  // a band's or DETAIL's raw b_k has no gain applied -- gain could be zero
  // -- and no fixed scale, so gating/scaling it here would be meaningless;
  // it gets its own rms-based normalization below instead. CORRECTION and
  // the real output both want the gate and master strength.
  if(!showing_texture)
  {
    // gain_local_contrast is a pure multiplier on top of whatever the bands
    // already summed to, so it belongs here rather than inside the ladder.
    DT_OMP_FOR()
    for(size_t k = 0; k < npixels; k++)
    {
      const float gate = _wiener_gate(coarsest[k], d->noise_bias);
      correction[k] *= gate * d->gain_local_contrast;
    }
  }

  if(showing_mask)
  {
    float divisor = 1.0f;
    if(showing_texture)
    {
      // scale by the displayed buffer's own frame-wide rms (research.md
      // §4.2) -- without it a low-amplitude band reads as flat grey.
      double sum_sq = 0.0;
      DT_OMP_FOR(reduction(+ : sum_sq))
      for(size_t k = 0; k < npixels; k++)
        sum_sq += (double)correction[k] * (double)correction[k];
      const float rms = sqrtf((float)(sum_sq / (double)npixels));
      divisor = fmaxf(3.0f * rms, 1e-3f);
    }

    dt_iop_gui_enter_critical_section(self);
    g->mask_divisor = divisor;
    dt_iop_gui_leave_critical_section(self);

    display_correction_mask(correction, divisor, out, roi_in);
    piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
  }
  else
  {
    apply_correction(in, correction, out, roi_in);
  }

  dt_free_align(luminance);
  dt_free_align(correction);
  dt_free_align(coarsest);
}

// dt_iop_contrast_data_t (sigma[]/gain[]/nbands on top of the params it
// copies) is bigger than dt_iop_contrast_params_t, but without an init_pipe
// of its own the module got develop/imageop.c's default_init_pipe, which
// sizes piece->data at self->params_size -- too small. commit_params below
// (which the module *does* override) then writes past the end of that
// undersized allocation on every call. Silent until whatever happens to sit
// next on the heap gets its chunk header corrupted, then a crash at a much
// later, unrelated free(): found by exporting a real image with the module
// enabled (every band gain != 1, so commit_params actually runs) under gdb --
// SIGABRT in dt_dev_pixelpipe_cleanup_nodes, malloc_printerr "free(): invalid
// next size". Needs its own init_pipe/cleanup_pipe sized to the real struct,
// same pattern as toneequal.c/colorequal.c/atrous.c.
void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_calloc1_align_type(dt_iop_contrast_data_t);
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
}

void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out,
                   dt_iop_roi_t *roi_in)
{
  dt_iop_contrast_data_t *const d = piece->data;

  // node k's nominal wavelength is S * 2^-(D0+k+shift); boundary k sits half
  // an octave finer, at the geometric mean of nodes k and k+1 -- see
  // implementation-plan.md §1.2. Walk from the finest node to the coarsest so
  // that dropping unresolvable bands (sigma < 0.7px at this roi scale) always
  // trims off the front and the last band processed is always the coarsest,
  // which never gets dropped.
  const float S = MAX(piece->iwidth, piece->iheight);
  int nbands = 0;
  for(int k = CT_BANDS - 1; k >= 0; k--)
  {
    const float D = CT_BAND_D0 + k + 0.5f + d->scale_shift;
    const float diameter = exp2f(-D) * S * roi_in->scale;
    const float sigma = 0.5f * (diameter - 1.0f);

    if(nbands == 0 && sigma < 0.7f) continue;  // unresolvable fine tail: drop

    d->sigma[nbands] = sigma;
    d->gain[nbands] = d->band[k];
    nbands++;
  }
  d->nbands = nbands;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_contrast_params_t *p = (dt_iop_contrast_params_t *)p1;
  dt_iop_contrast_data_t *d = piece->data;

  d->iterations = p->filter_iterations;
  d->gain_local_contrast = p->gain_local_contrast;
  d->noise_bias = p->noise_bias;
  d->scale_shift = p->scale_shift;
  d->decomposition = p->decomposition;
  for(int k = 0; k < CT_BANDS; k++)
    d->band[k] = p->band[k];

  // UI feathering is inverted (higher = stricter edge preservation).
  // Adjust the strength based on the number of iterations to maintain a consistent overall effect regardless of iteration count.
  const float default_feathering = 0.2f;  // Base value based on Christian's experiments for a good balance of edge preservation and contrast boost at default settings.
  d->feathering = default_feathering * powf(2.0f, -p->edge_protection) / (p->filter_iterations * p->filter_iterations);
}

// redraw the graph once a pipe has actually run, so its stripe shading
// (g->nbands, published from process() above) and node positions track
// what just got computed rather than the last GUI edit (§1.5; atrous.c's
// _ui_pipe_done does the same for its own frequency histogram).
static void _ui_pipe_done(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(g && !DT_IN_GUI_UPDATE() && self->enabled && self->expanded)
    gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

// §2.2: query the ladder's published SAT tables for the box the picker
// selected, and feed the resulting per-rung (wavelength, energy, weight)
// triples to §2.3's `_fit_spectrum` -- primed, per §2.4, with a frame-wide
// noise estimate rather than fitting N freely -- and (§2.4) read the box's
// own sparseness back out of the same tables. box is in the pixels of
// g->ladder_roi_in, i.e. of the preview the ladder was built from. returns
// the fitted texture wavelength in those same pixels, or 0 if nothing was
// measurable in the box -- see below for what "nothing" covers now that the
// fit separates noise and self-similar content from an actual texture size.
//
// research.md §5.2: any box query is 4 lookups per rung -- this is that
// query, one held critical section covering every rung so the buffer can't
// be resized out from under it mid-query.
static double _fit_curve_from_box(dt_iop_module_t *self, const int *const box)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  double lambda[CT_MAX_BANDS], energies[CT_MAX_BANDS], weights[CT_MAX_BANDS];
  double s1_energy[CT_MAX_BANDS];    // §2.4: Sum(|b|)/n_eff over the box, for its own sparseness
  double noise_floor[CT_MAX_BANDS];  // §2.4: frame-wide, not the box's own
  int nrungs = 0;

  dt_iop_gui_enter_critical_section(self);

  const size_t sat_w = g->pd.width, sat_h = g->pd.height;
  const size_t comps = g->pd.components;
  const gboolean have_data =
    g->pd.buf && sat_w > 1 && sat_h > 1 && g->ladder_nrungs > 0
    && comps == (size_t)(2 * g->ladder_nrungs);

  if(have_data)
  {
    const size_t bw = sat_w - 1, bh = sat_h - 1;
    const size_t bx0 = MIN(bw, (size_t)MAX(box[0], 0) / CT_BLOCK);
    size_t bx1 = MIN(bw, (size_t)(MAX(box[2], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(bx1 <= bx0) bx1 = MIN(bw, bx0 + 1);
    const size_t by0 = MIN(bh, (size_t)MAX(box[1], 0) / CT_BLOCK);
    size_t by1 = MIN(bh, (size_t)(MAX(box[3], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(by1 <= by0) by1 = MIN(bh, by0 + 1);
    const double nblocks = (double)(bx1 - bx0) * (double)(by1 - by0);

    nrungs = g->ladder_nrungs;
    const float *const restrict buf = g->pd.buf;
    for(int r = 0; r < nrungs; r++)
    {
      const double s2 = buf[(by1 * sat_w + bx1) * comps + 2 * r]
                       - buf[(by0 * sat_w + bx1) * comps + 2 * r]
                       - buf[(by1 * sat_w + bx0) * comps + 2 * r]
                       + buf[(by0 * sat_w + bx0) * comps + 2 * r];
      const double s1 = buf[(by1 * sat_w + bx1) * comps + 2 * r + 1]
                       - buf[(by0 * sat_w + bx1) * comps + 2 * r + 1]
                       - buf[(by1 * sat_w + bx0) * comps + 2 * r + 1]
                       + buf[(by0 * sat_w + bx0) * comps + 2 * r + 1];

      // n_per_block is deterministic from the rung's own decimation: CT_BLOCK
      // level-0 pixels per block side, step level-0 pixels per level pixel of
      // this rung, so (CT_BLOCK/step)^2 level pixels per block once the
      // ladder hasn't decimated past CT_BLOCK yet, one (correlated) source
      // pixel spread over several blocks once it has.
      const double step = g->ladder_step[r];
      const double n_per_block = fmax(1.0, (double)(CT_BLOCK * CT_BLOCK) / (step * step));
      const double n_eff = fmax(nblocks * n_per_block, 1.0);

      lambda[r] = g->ladder_lambda[r];
      energies[r] = s2 / n_eff;
      s1_energy[r] = s1 / n_eff;
      weights[r] = 1.0 / (CT_MODEL_ERROR * CT_MODEL_ERROR + 2.0 / n_eff);
      noise_floor[r] = g->ladder_noise_floor[r];
    }
  }

  dt_iop_gui_leave_critical_section(self);

  if(!have_data) return 0.0;

  double peak_e = 0.0;
  for(int r = 0; r < nrungs; r++) peak_e = fmax(peak_e, energies[r]);

  // §2.4: fix N from the frame-wide block-minimum estimate rather than
  // fitting it freely -- "stabilises everything else" (research.md §5.5) and
  // stops a genuinely fine texture from getting explained away as noise.
  const double noise_prior = _ladder_estimate_noise(lambda, noise_floor, nrungs);

  _ct_fit_t fit;
  if(!_fit_spectrum(lambda, energies, weights, nrungs, noise_prior, &fit)) return 0.0;

  // §2.4/research.md §5.9: advisory only -- does not refuse the pick, just
  // explains a result that might otherwise look like nothing happened.
  // evaluated at the box's own peak-energy rung: S(lambda) = A*G(lambda;tau)
  // + C*lambda^(beta-2) is what the fit calls real content, N(lambda) =
  // N*G(lambda;0) is what it calls noise, both in the same (s = sigma^2)
  // basis _fit_spectrum solved in.
  {
    int peak_idx = 0;
    for(int r = 1; r < nrungs; r++) if(energies[r] > energies[peak_idx]) peak_idx = r;
    const double sigma_pk = lambda[peak_idx] / (2.0 * M_PI);
    const double s_pk = sigma_pk * sigma_pk;
    const double S = fit.texture * _dog_shape(s_pk, fit.tau)
                    + fit.self_similar * pow(s_pk, (fit.beta - 2.0) * 0.5);
    const double N = fit.noise * _dog_shape(s_pk, 0.0);
    if(S <= (S + N) * CT_NOISE_DOMINATED_FRAC)
      dt_control_log(_("the picked area looks like noise -- try raising the noise bias"));
  }

  // what the texture term actually delivers over the measured rungs
  // (fit.texture_peak) is negligible next to the box's own peak energy:
  // there is no sized texture here to report a wavelength for, only sensor
  // noise and/or ordinary (beta != 2) scene content -- comparing fit.texture
  // itself to fit.noise/fit.self_similar would not mean anything, since
  // _dog_shape's amplitude and the self-similar power law's live on
  // unrelated scales. research.md §5.6 (Phase 2.5) is what turns "no real
  // texture" into a proper fallback to a "detail" target curve; until then,
  // fall back the same way `_fit_texture_scale` used to on a self-similar
  // area -- report the finest measurable size, which boosts what is there
  // without reaching for structure that isn't, rather than declining
  // outright.
  if(fit.texture_peak <= peak_e * 1e-2) return lambda[0];

  // §2.4/research.md §5.9: advisory only, does not refuse the pick -- warns
  // that the measured size may not mean much when the box's rung nearest the
  // fitted texture size looks sparse (kappa >> CT_KAPPA_GAUSSIAN) rather than
  // dense, i.e. more like one strong edge crossing the box than real texture.
  {
    const double target_lambda = 2.0 * M_PI * sqrt(fit.tau);
    int nearest = 0;
    double best_d = DBL_MAX;
    for(int r = 0; r < nrungs; r++)
    {
      const double dist = fabs(log(lambda[r] / target_lambda));
      if(dist < best_d) { best_d = dist; nearest = r; }
    }
    if(s1_energy[nearest] > 0.0)
    {
      const double kappa = sqrt(energies[nearest]) / s1_energy[nearest];
      if(kappa > CT_KAPPA_EDGE)
        dt_control_log(_("the picked area looks more like a hard edge than dense texture -- "
                          "the measured size may be unreliable"));
    }
  }

  // the hump peaks at t = 2*tau, i.e. at sigma = sqrt(2)*sigma_t, and the
  // normalized Laplacian of a sinusoid of wavelength L peaks at
  // sigma = L / (pi*sqrt(2)) -- so the wavelength this stands for is simply
  // 2*pi*sigma_t (`_fit_texture_scale`'s established convention).
  return 2.0 * M_PI * sqrt(fit.tau);
}

// §2.2: synchronous now that the ladder is frame-wide and pre-published
// (§2.1/§2.2 above) -- no more arming a pick and waiting for a preview pass
// to claim and measure it (research.md §5.11). the box maps straight from
// the color picker's sample to the ladder's own roi, the SAT query is a
// handful of lookups, and the fit is a grid search over ~80 points: all fast
// enough to run inline on the GUI thread instead of round-tripping through
// another preview pass.
void color_picker_apply(dt_iop_module_t *self,
                        GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(!g || picker != g->scale_shift) return;

  if(!dt_preview_data_is_fresh(&g->pd))
  {
    dt_control_log(_("wait for the preview to finish recomputing"));
    return;
  }

  dt_iop_gui_enter_critical_section(self);
  const dt_iop_roi_t roi_in = g->ladder_roi_in;
  dt_iop_gui_leave_critical_section(self);

  // region defaults to the whole frame; a box pick narrows it, same
  // fallback picked-region measurement always used.
  int box[4] = { 0, 0, (int)roi_in.width, (int)roi_in.height };
  const dt_colorpicker_sample_t *const sample =
    darktable.lib->proxy.colorpicker.primary_sample;
  if(sample && sample->size == DT_LIB_COLORPICKER_SIZE_BOX)
  {
    int picked[4];
    if(!dt_color_picker_box(self, &roi_in, sample, PIXELPIPE_PICKER_INPUT, picked))
      memcpy(box, picked, sizeof(box));
  }

  const double wavelength = _fit_curve_from_box(self, box);
  if(wavelength <= 0.0)
  {
    dt_control_log(_("the picked area is too small, or has nothing in it to measure a detail size from"));
    return;
  }

  // invert what modify_roi_in() does with a node's placement: node D's
  // window is 2^-D * max_size * roi->scale pixels wide, so asking for a
  // window exactly one texture wavelength wide -- the shortest box average
  // that removes that texture from the base layer completely, and so hands
  // all of it to the high pass without also dragging in anything coarser --
  // gives the D below, which is folded into scale_shift.
  //
  // the scale factor cancels the fact that this was measured on the preview:
  // what comes out is a fraction of the frame and is carried unchanged to
  // full resolution. what does not cancel is that the preview cannot resolve
  // texture finer than a few of its own pixels, which is what bounds the
  // fine end of the answer.
  const double max_size = MAX(pipe->iwidth, pipe->iheight);
  const double scale = fmax((double)roi_in.scale, 1e-6);
  const float level =
    (float)CLAMP(log2(max_size * scale / wavelength), 0.0, 15.0);

  // write into params and commit *before* refreshing the widget: the refresh
  // below runs under the gui-update guard and so deliberately writes nothing
  // back, which is the whole point of that guard -- it normally runs the other
  // way around, syncing widgets to params that have already changed.
  dt_iop_contrast_params_t *p = self->params;
  const float d = CLAMP(level, CT_BAND_D0, CT_BAND_D0 + CT_BANDS - 1);
  p->scale_shift = CLAMP(d - roundf(d), -0.5f, 0.5f);

  // §1.7: reshape the bands into a single hump centred on the measured size,
  // so the pick does something visible beyond repositioning the ladder. a
  // hump peaking at the neutral gain would be invisible, so if the master
  // gain hasn't been touched yet, raise it first.
  if(p->gain_local_contrast == 1.0f) p->gain_local_contrast = 1.5f;
  const float center = d - CT_BAND_D0;  // continuous band-index units
  const float width = 1.0f;             // octaves either side of the peak
  for(int k = 0; k < CT_BANDS; k++)
  {
    const float dist = (k - center) / width;
    p->band[k] = 1.0f + (p->gain_local_contrast - 1.0f) * expf(-0.5f * dist * dist);
  }

  dt_dev_add_history_item(darktable.develop, self, TRUE);

  // pull the bound widgets back from params by hand rather than through
  // dt_iop_gui_update(): that also refreshes the blending UI, which switches
  // the color picker off outright (keep = FALSE) whenever blending is not in
  // parametric mode -- and the picker that asked for this measurement is meant
  // to stay armed so the area can be dragged again.
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_update_from_field(self, NULL, NULL, NULL);
  DT_LEAVE_GUI_UPDATE();
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(in) return;

  dt_iop_color_picker_reset(self, TRUE);
}

// shared by every quad (§1.6): early return if the blend module is already
// displaying a mask -- unchanged from the module's original single-mode
// check, now called from both the slider quads and the graph's ctrl+click.
static gboolean _mask_display_blocked(dt_iop_module_t *self)
{
  if(!self->request_mask_display) return FALSE;
  dt_control_log(_("cannot display masks when the blending mask is displayed"));
  return TRUE;
}

// sets g->details_display (toggling it off if it was already showing mode)
// and keeps every quad's active state in sync with it -- exactly one active
// at a time, matching what process() will actually display next pass.
static void _apply_details_display(dt_iop_module_t *self, const dt_iop_details_display_t mode)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_iop_request_focus(self);
  // Activate the module if it wasn't
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  g->details_display = (g->details_display == mode) ? DT_CT_MASK_OFF : mode;

  dt_bauhaus_widget_set_quad_active
    (GTK_WIDGET(g->gain_local_contrast),
     g->details_display == DT_CT_MASK_CORRECTION || g->details_display == DT_CT_MASK_DETAIL);
  for(int k = 0; k < CT_BANDS; k++)
    dt_bauhaus_widget_set_quad_active(GTK_WIDGET(g->band[k]), g->details_display == k);

  dt_iop_refresh_center(self);
}

// the master gain's quad shows CORRECTION (ctrl+click: DETAIL); each band
// slider's quad shows that band's own raw texture ("ct-band" data set at
// creation, gui_init).
static void show_details_callback(GtkWidget *togglebutton, dt_iop_module_t *self)
{
  if(_mask_display_blocked(self))
  {
    dt_bauhaus_widget_set_quad_active(GTK_WIDGET(togglebutton), FALSE);
    return;
  }

  DT_GUARD_GUI_UPDATE();

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_iop_details_display_t mode;
  if(togglebutton == g->gain_local_contrast)
    mode = dt_modifier_is(dt_key_modifier_state(), GDK_CONTROL_MASK)
      ? DT_CT_MASK_DETAIL : DT_CT_MASK_CORRECTION;
  else
    mode = (dt_iop_details_display_t)GPOINTER_TO_INT(g_object_get_data(G_OBJECT(togglebutton), "ct-band"));

  _apply_details_display(self, mode);
}

// ---------------------------------------------------------------------------
// the graph (implementation-plan.md §1.4)
// ---------------------------------------------------------------------------
//
// nodes run coarse (left) to fine (right), evenly spaced -- one per octave,
// which is exactly what CT_BANDS is -- so a node's x fraction is simply
// (k + 0.5) / CT_BANDS and needs no lookup. y is gain, soft-ranged to
// CT_GRAPH_Y_MAX to match the sliders (gui_init); dragging above the visible
// top still reaches the sliders' hard max, same as overdriving a slider past
// its soft range.
//
// every handler below re-derives the graph's pixel geometry from the
// widget's current allocation rather than caching it, which is what keeps a
// resize from desyncing the nodes (§1.4 acceptance).

static void _graph_curve_from_params(dt_draw_curve_t *curve,
                                     const dt_iop_contrast_params_t *const p)
{
  for(int k = 0; k < CT_BANDS; k++)
    dt_draw_curve_set_point(curve, k, (k + 0.5f) / (float)CT_BANDS,
                            CLAMP(p->band[k] / CT_GRAPH_Y_MAX, 0.0f, 1.0f));
}

static void _graph_geometry(GtkWidget *widget, int *inset, int *width, int *height)
{
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  *inset = DT_PIXEL_APPLY_DPI(4);
  *width = allocation.width - 2 * (*inset);
  *height = allocation.height - 2 * (*inset) - DT_RESIZE_HANDLE_SIZE;
}

static int _graph_band_at(const int width, const double x)
{
  const int k = (int)floor(x / (double)MAX(width, 1) * CT_BANDS);
  return CLAMP(k, 0, CT_BANDS - 1);
}

// inverse of the node-drawing map in _area_draw: pixel y (0 at the graph's
// top) to a gain. left unclamped to CT_GRAPH_Y_MAX on purpose -- dragging
// above the visible top keeps climbing, all the way to the slider's own hard
// range, exactly like overdriving a slider past its soft range.
static float _graph_gain_at(const int height, const double y)
{
  const float yfrac = 1.0f - (float)(y / (double)MAX(height, 1));
  return CLAMP(yfrac * CT_GRAPH_Y_MAX, 0.0f, 5.0f);
}

static void _area_set_band(dt_iop_contrast_gui_data_t *g, const int k, const float gain)
{
  if(k < 0 || k >= CT_BANDS || !g->band[k]) return;
  dt_bauhaus_slider_set_val(g->band[k], gain);
}

static gboolean _area_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  const dt_iop_contrast_params_t *const p = self->params;

  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  if(width <= 0 || height <= 0) return FALSE;

  gtk_widget_set_tooltip_text
    (widget,
     g->nbands < CT_BANDS
     ? _("drag a node to set its band's gain; double-click to reset it;\n"
         "ctrl+click to visualize that band's own detail texture;\n"
         "middle-click for the plain slider list.\n"
         "the shaded bands on the right are too fine to resolve at the\n"
         "current zoom level and have no effect until you zoom in.")
     : _("drag a node to set its band's gain; double-click to reset it;\n"
         "ctrl+click to visualize that band's own detail texture;\n"
         "middle-click for the plain slider list."));

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);

  cairo_surface_t *cst =
    dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, allocation.width, allocation.height);
  cairo_t *cr = cairo_create(cst);

  GtkStyleContext *context = gtk_widget_get_style_context(widget);
  gtk_render_background(context, cr, 0, 0, allocation.width, allocation.height);
  cairo_translate(cr, inset, inset);

  // 1. background grid: one line per octave, i.e. per node
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));
  set_color(cr, darktable.bauhaus->graph_border);
  dt_draw_grid(cr, CT_BANDS, 0, 0, width, height);

  // 2. unresolvable-band shading -- bands beyond g->nbands (§1.5) don't
  // survive the current pipe scale and have no effect
  if(g->nbands < CT_BANDS)
  {
    const float x0 = (float)g->nbands / (float)CT_BANDS * width;
    cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                             darktable.bauhaus->graph_border.green,
                             darktable.bauhaus->graph_border.blue, 0.4);
    cairo_rectangle(cr, x0, 0, width - x0, height);
    cairo_fill(cr);
  }

  // 3. baseline at gain 1.0
  const float baseline_y = height * (1.0f - 1.0f / CT_GRAPH_Y_MAX);
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
  dt_draw_line(cr, 0, baseline_y, width, baseline_y);
  cairo_stroke(cr);

  // 5. the curve: monotone cubic through the nine nodes (research.md's
  // Phase 2 spectrum overlay is not built yet, so there is no step 4 here)
  _graph_curve_from_params(g->curve, p);
  float xs[CT_GRAPH_RES], ys[CT_GRAPH_RES];
  dt_draw_curve_calc_values(g->curve, 0.0f, 1.0f, CT_GRAPH_RES, xs, ys);
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.0));
  cairo_move_to(cr, 0, height * (1.0f - ys[0]));
  for(int i = 1; i < CT_GRAPH_RES; i++)
    cairo_line_to(cr, i * width / (float)(CT_GRAPH_RES - 1), height * (1.0f - ys[i]));
  cairo_stroke(cr);

  // 6. node bars + bullets
  for(int k = 0; k < CT_BANDS; k++)
  {
    const float xn = (k + 0.5f) / CT_BANDS * width;
    const float yfrac = CLAMP(p->band[k] / CT_GRAPH_Y_MAX, 0.0f, 1.0f);
    const float yn = height * (1.0f - yfrac);

    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(6));
    set_color(cr, darktable.bauhaus->color_fill);
    dt_draw_line(cr, xn, baseline_y, xn, yn);
    cairo_stroke(cr);

    const gboolean active = (k == g->hover_band || k == g->drag_band);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
    cairo_arc(cr, xn, yn, DT_PIXEL_APPLY_DPI(active ? 5.0 : 3.5), 0.0, 2.0 * M_PI);
    set_color(cr, darktable.bauhaus->graph_fg);
    cairo_stroke_preserve(cr);
    if(k == g->drag_band)
      set_color(cr, darktable.bauhaus->graph_fg);
    else
      set_color(cr, darktable.bauhaus->graph_bg);
    cairo_fill(cr);

    // §1.6: this band's own texture is the mask currently shown -- the link
    // that answers "you can only visualize the main detail scale"
    if(k == g->details_display)
    {
      cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
      set_color(cr, darktable.bauhaus->graph_fg);
      cairo_arc(cr, xn, yn, DT_PIXEL_APPLY_DPI(7.5), 0.0, 2.0 * M_PI);
      cairo_stroke(cr);
    }
  }

  // §1.6: print the divisor a texture mask view was normalized by, so the
  // view stays quantitative rather than just "brighter means more"
  if(g->details_display == DT_CT_MASK_DETAIL
     || (g->details_display >= 0 && g->details_display < CT_BANDS))
  {
    dt_iop_gui_enter_critical_section(self);
    const float divisor = g->mask_divisor;
    dt_iop_gui_leave_critical_section(self);

    char buf[32];
    snprintf(buf, sizeof(buf), "×%.4g", (double)divisor);
    PangoFontDescription *div_desc = dt_gui_get_font();
    pango_font_description_set_absolute_size(div_desc, 0.09 * height * PANGO_SCALE);
    PangoLayout *div_layout = pango_cairo_create_layout(cr);
    pango_layout_set_font_description(div_layout, div_desc);
    set_color(cr, darktable.bauhaus->graph_fg);
    pango_layout_set_text(div_layout, buf, -1);
    PangoRectangle div_ink;
    pango_layout_get_pixel_extents(div_layout, &div_ink, NULL);
    cairo_move_to(cr, width - div_ink.width - DT_PIXEL_APPLY_DPI(2),
                 height - div_ink.height - DT_PIXEL_APPLY_DPI(2));
    pango_cairo_show_layout(cr, div_layout);
    g_object_unref(div_layout);
    pango_font_description_free(div_desc);
  }

  // axis labels
  PangoFontDescription *desc = dt_gui_get_font();
  pango_font_description_set_absolute_size(desc, 0.09 * height * PANGO_SCALE);
  PangoLayout *layout = pango_cairo_create_layout(cr);
  pango_layout_set_font_description(layout, desc);
  set_color(cr, darktable.bauhaus->graph_fg);

  pango_layout_set_text(layout, _("coarse"), -1);
  cairo_move_to(cr, DT_PIXEL_APPLY_DPI(2), DT_PIXEL_APPLY_DPI(2));
  pango_cairo_show_layout(cr, layout);

  PangoRectangle ink;
  pango_layout_set_text(layout, _("fine"), -1);
  pango_layout_get_pixel_extents(layout, &ink, NULL);
  cairo_move_to(cr, width - ink.width - DT_PIXEL_APPLY_DPI(2), DT_PIXEL_APPLY_DPI(2));
  pango_cairo_show_layout(cr, layout);

  g_object_unref(layout);
  pango_font_description_free(desc);

  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return FALSE;
}

static void _area_motion(GtkEventControllerMotion *controller,
                         gdouble x, gdouble y,
                         dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  GtkWidget *widget = dt_gui_get_widget(controller);
  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  const double gx = x - inset, gy = y - inset;

  g->hover_band = (gx >= 0 && gx <= width) ? _graph_band_at(width, gx) : -1;

  if(g->dragging && g->drag_band >= 0)
    _area_set_band(g, g->drag_band, _graph_gain_at(height, gy));

  gtk_widget_queue_draw(widget);
}

static void _area_leave(GtkEventControllerMotion *controller, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  g->hover_band = -1;
  gtk_widget_queue_draw(dt_gui_get_widget(controller));
}

static void _area_button_press(GtkGestureSingle *gesture,
                               gint n_press,
                               gdouble x, gdouble y,
                               dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  GtkWidget *widget = dt_gui_get_widget(gesture);
  const guint button = gtk_gesture_single_get_current_button(gesture);

  if(button == GDK_BUTTON_MIDDLE)
  {
    const gchar *current = gtk_stack_get_visible_child_name(g->stack);
    gtk_stack_set_visible_child_name(g->stack,
                                     g_strcmp0(current, "graph") == 0 ? "sliders" : "graph");
    return;
  }

  if(button != GDK_BUTTON_PRIMARY) return;

  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  const int k = _graph_band_at(width, x - inset);

  // ctrl+click selects that band's mask view (§1.6) rather than dragging its
  // value -- the two would otherwise fight over the same click.
  if(dt_modifier_is(dt_key_modifier_state(), GDK_CONTROL_MASK))
  {
    if(!_mask_display_blocked(self))
      _apply_details_display(self, (dt_iop_details_display_t)k);
    gtk_widget_queue_draw(widget);
    return;
  }

  if(n_press >= 2)
  {
    const dt_iop_contrast_params_t *const def = self->default_params;
    _area_set_band(g, k, def->band[k]);
    return;
  }

  g->dragging = TRUE;
  g->drag_band = k;
  _area_set_band(g, k, _graph_gain_at(height, y - inset));
  gtk_widget_queue_draw(widget);
}

static void _area_button_release(GtkGestureSingle *gesture,
                                 gint n_press,
                                 gdouble x, gdouble y,
                                 dt_iop_module_t *self)
{
  if(gtk_gesture_single_get_current_button(gesture) != GDK_BUTTON_PRIMARY) return;
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  g->dragging = FALSE;
  g->drag_band = -1;
  gtk_widget_queue_draw(dt_gui_get_widget(gesture));
}

static void _area_scrolled(GtkEventControllerScroll *controller,
                           gdouble dx, gdouble dy,
                           dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(g->hover_band < 0 || dy == 0.0) return;

  const dt_iop_contrast_params_t *const p = self->params;
  const float step = dt_modifier_eq(controller, GDK_CONTROL_MASK) ? 0.01f : 0.05f;
  _area_set_band(g, g->hover_band, p->band[g->hover_band] - (float)dy * step);
}

enum
{
  DT_ACTION_EFFECT_CT_RESET = DT_ACTION_EFFECT_RESET,
};

static const dt_action_element_def_t _action_elements_ct[]
  = { { N_("band 1 (coarsest)"), dt_action_effect_value },
      { N_("band 2"), dt_action_effect_value },
      { N_("band 3"), dt_action_effect_value },
      { N_("band 4"), dt_action_effect_value },
      { N_("band 5"), dt_action_effect_value },
      { N_("band 6"), dt_action_effect_value },
      { N_("band 7"), dt_action_effect_value },
      { N_("band 8"), dt_action_effect_value },
      { N_("band 9 (finest)"), dt_action_effect_value },
      { } };

static float _action_process_ct(gpointer target,
                                const dt_action_element_t element,
                                const dt_action_effect_t effect,
                                float move_size)
{
  if(element < 0 || element >= CT_BANDS) return DT_ACTION_NOT_VALID;

  dt_iop_module_t *self = g_object_get_data(G_OBJECT(target), "iop-instance");
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_iop_contrast_params_t *p = self->params;
  const dt_iop_contrast_params_t *const d = self->default_params;

  if(DT_PERFORM_ACTION(move_size))
  {
    switch(effect)
    {
      case DT_ACTION_EFFECT_CT_RESET:
        _area_set_band(g, element, d->band[element]);
        break;
      case DT_ACTION_EFFECT_DOWN:
        move_size *= -1;
      case DT_ACTION_EFFECT_UP:
        _area_set_band(g, element, p->band[element] + move_size / 100.0f);
        break;
      default:
        break;
    }
    gtk_widget_queue_draw(GTK_WIDGET(g->area));
  }

  return p->band[element] + DT_VALUE_PATTERN_PLUS_MINUS;
}

static const dt_action_def_t _action_def_ct
  = { N_("detail ladder"),
      _action_process_ct,
      _action_elements_ct,
      NULL };

void gui_init(dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = IOP_GUI_ALLOC(contrast);
  g->details_display = DT_CT_MASK_OFF;
  g->mask_divisor = 1.0f;
  dt_preview_data_alloc(&g->pd, self);  // §2.2: the frame-wide ladder's SAT tables

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_UI_PIPE_FINISHED, _ui_pipe_done);

  // Main container
  self->widget = dt_gui_vbox();

  // Local boost slider
  g->gain_local_contrast = dt_bauhaus_slider_from_params(self, "gain_local_contrast");
  dt_bauhaus_slider_set_soft_range(g->gain_local_contrast, 0.0, 2.0);
  dt_bauhaus_slider_set_digits(g->gain_local_contrast, 2);
  dt_bauhaus_slider_set_format(g->gain_local_contrast, "%");
  dt_bauhaus_slider_set_factor(g->gain_local_contrast, 100.0);
  dt_bauhaus_slider_set_offset(g->gain_local_contrast, -100.0);
  gtk_widget_set_tooltip_text(g->gain_local_contrast,
                              _("amount of local contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->gain_local_contrast, self, dtgtk_cairo_paint_showmask, TRUE, show_details_callback,
                             _("visualize the accumulated correction -- what this module is doing.\n"
                               "ctrl+click: visualize the raw, un-gained detail sum (v1's behaviour)."));

  // Filter settings section
  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "filter settings")));

  // the graph (§1.4): nine nodes, one per octave, drawn as a curve and
  // draggable directly; a GtkStack toggled by middle-click on the graph
  // swaps it for the plain slider list, which is what the shortcut system
  // and anyone chasing an exact value still reach for.
  g->nbands = CT_BANDS;  // until the first preview pass publishes the real count (§1.5)
  g->hover_band = -1;
  g->drag_band = -1;
  g->dragging = FALSE;
  g->curve = dt_draw_curve_new(0.0, 1.0, MONOTONE_HERMITE);
  for(int k = 0; k < CT_BANDS; k++)
    dt_draw_curve_add_point(g->curve, (k + 0.5f) / (float)CT_BANDS,
                            CLAMP(((dt_iop_contrast_params_t *)self->default_params)->band[k]
                                  / CT_GRAPH_Y_MAX, 0.0f, 1.0f));

  g->area = GTK_DRAWING_AREA(dt_ui_resize_wrap
                             (NULL, 0, "plugins/darkroom/contrastadv/graphheight"));
  g_object_set_data(G_OBJECT(g->area), "iop-instance", self);
  dt_action_define_iop(self, NULL, N_("graph"), GTK_WIDGET(g->area), &_action_def_ct);
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_area_draw), self);
  dt_gui_connect_click(g->area, _area_button_press, _area_button_release, self);
  dt_gui_connect_motion(g->area, _area_motion, _area_motion, _area_leave, self);
  dt_gui_connect_scroll(g->area, GTK_EVENT_CONTROLLER_SCROLL_BOTH_AXES
                               | GTK_EVENT_CONTROLLER_SCROLL_DISCRETE, _area_scrolled, self);

  // one slider per band, labeled by the node's nominal size -- computed
  // rather than nine near-identical translated strings, per
  // implementation-plan.md §1.1.
  GtkWidget *sliders_box = dt_gui_vbox();
  dt_iop_module_t *section = DT_IOP_SECTION_FOR_PARAMS(self, NC_("section", "bands"), sliders_box);
  for(int k = 0; k < CT_BANDS; k++)
  {
    char param[16];
    snprintf(param, sizeof(param), "band[%d]", k);
    g->band[k] = dt_bauhaus_slider_from_params(section, param);
    dt_bauhaus_slider_set_soft_range(g->band[k], 0.0, 2.0);
    dt_bauhaus_slider_set_digits(g->band[k], 2);
    dt_bauhaus_slider_set_format(g->band[k], "%");
    dt_bauhaus_slider_set_factor(g->band[k], 100.0);
    dt_bauhaus_slider_set_offset(g->band[k], -100.0);

    char label[64];
    snprintf(label, sizeof(label), _("detail size ~ %.3g%%"), 100.0 * exp2(-(CT_BAND_D0 + k)));
    dt_bauhaus_widget_set_label(g->band[k], NULL, label);

    // §1.6: this quad shows this band's own raw detail texture, whatever its
    // gain -- "ct-band" is what show_details_callback reads to know which.
    g_object_set_data(G_OBJECT(g->band[k]), "ct-band", GINT_TO_POINTER(k));
    dt_bauhaus_widget_set_quad(g->band[k], self, dtgtk_cairo_paint_showmask, TRUE, show_details_callback,
                               _("visualize this band's own detail texture\n"
                                 "(the same as ctrl+clicking its node in the graph)"));
  }

  g->stack = GTK_STACK(gtk_stack_new());
  gtk_stack_set_homogeneous(g->stack, FALSE);
  gtk_stack_add_named(g->stack, GTK_WIDGET(g->area), "graph");
  gtk_stack_add_named(g->stack, sliders_box, "sliders");
  gtk_stack_set_visible_child_name(g->stack, "graph");
  dt_action_define_iop(self, NULL, N_("sliders"), GTK_WIDGET(g->stack), NULL);
  dt_gui_box_add(self->widget, g->stack);

  g->scale_shift = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                       dt_bauhaus_slider_from_params(self, "scale_shift"));
  gtk_widget_set_tooltip_text(g->scale_shift,
     _("shifts every band's node together, finer or coarser.\n"
       "half a step moves the whole ladder by half an octave."));
  dt_bauhaus_widget_set_quad_tooltip
    (g->scale_shift,
     _("pick an area: measure how large the texture in it actually is, and\n"
       "shift the ladder so a node lands on that size.\n"
       "click to use the whole frame, then drag on the image to work from the\n"
       "subject that matters instead. a small box cannot report structure\n"
       "larger than itself, so pick over as much of the texture as you want\n"
       "counted.\n"
       "an area with nothing in it to enhance -- clear sky, an out-of-focus\n"
       "background -- is declined rather than guessed at. if a pick over deep\n"
       "shadow comes back at the finest setting it is reading sensor noise;\n"
       "raise the noise bias below and pick again."));

  g->decomposition = dt_bauhaus_combobox_from_params(self, "decomposition");
  gtk_widget_set_tooltip_text(g->decomposition,
     _("accurate: every band a direct, full-resolution pass. the default.\n"
       "fast: the coarsest bands run on a lower-resolution pyramid instead,\n"
       "about 30% cheaper. can soften dense fine texture (fur, hair) very\n"
       "slightly -- leave off unless you need the speed."));

  g->edge_protection = dt_bauhaus_slider_from_params(self, "edge_protection");
  dt_bauhaus_slider_set_soft_range(g->edge_protection, -2.0, 2.0);
  dt_bauhaus_slider_set_digits(g->edge_protection, 2);
  dt_bauhaus_slider_set_format(g->edge_protection, "%");
  dt_bauhaus_slider_set_factor(g->edge_protection, 100.0);
  gtk_widget_set_tooltip_text(g->edge_protection, _("adjust the edge sensitivity of the filter\n"
                                                    "higher = more edge preservation\n"
                                                    "lower = smoother transitions, but may lead to halos around edges"));

  g->filter_iterations = dt_bauhaus_slider_from_params(self, "filter_iterations");
  dt_bauhaus_slider_set_soft_range(g->filter_iterations, 1, 5);
  gtk_widget_set_tooltip_text(g->filter_iterations, _("number of passes of the guided filter to apply\n"
       "helps diffusing the edges of the filter at the expense of speed"));

  g->noise_bias = dt_bauhaus_slider_from_params(self, "noise_bias");
  dt_bauhaus_slider_set_soft_range(g->noise_bias, 0.0, 0.2);
  dt_bauhaus_slider_set_digits(g->noise_bias, 4);
  dt_bauhaus_slider_set_step(g->noise_bias, 0.0001);
  gtk_widget_set_tooltip_text(g->noise_bias, _("add bias to reduce shadow noise amplification.\n"
                                               "only affects dark parts of the image."));
}

// §1.8: dt_iop_gui_cleanup_module() (develop/imageop.c) already
// DT_CONTROL_SIGNAL_DISCONNECT_ALL()s every DT_CONTROL_SIGNAL_HANDLE
// registered in gui_init before calling this, so the ui-pipe-done handler
// needs no disconnect of its own here -- verified against the framework
// rather than assumed, and matches every other iop's gui_cleanup in this
// tree (atrous, toneequal).
void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;

  // drop any pick this module still owns, mirroring colorequal.c/toneequal.c
  // -- if the module is torn down while a pick is armed or in flight,
  // nothing else clears the global color picker for us.
  self->request_color_pick = DT_REQUEST_COLORPICK_OFF;

  dt_draw_curve_destroy(g->curve);
  dt_preview_data_free(&g->pd);  // §2.2
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
