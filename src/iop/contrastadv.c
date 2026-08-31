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

// §3.3/research.md §2.4: eps_k = eps * (sigma_k/sigma_ref)^p, sigma_ref the
// finest surviving band. p is deliberately small and un-exposed ("a small p,
// not another slider") -- 0.3 is a plausible starting point sized against
// phase0-band-energy.md's own numbers (raising the *global* eps from 0.2 to
// 0.8, a 4x change, was enough to keep every band alive there), not a
// value re-validated with phase0's own visual A/B method against real
// images; revisit if a future pass finds coarse bands still collapsing, or
// overshooting into halos, at this setting.
#define CT_FEATHERING_EXPONENT 0.3f

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

// implementation-plan-2.md §3.1: peak wavelength of the DoG between sigma and
// 2^(1/CT_SCALES_PER_OCTAVE)*sigma -- lambda = pi*sqrt(2*(k2-1)/ln k2) * sigma,
// k2 = 2^(2/CT_SCALES_PER_OCTAVE) -- _band_peak_lambda's formula specialised
// to the ladder's own fixed rung ratio. The single conversion factor between
// a rung's sigma and the wavelength label it is published under; recompute if
// CT_SCALES_PER_OCTAVE ever changes.
#define CT_SIGMA_TO_LAMBDA 5.0091626

// §2.1: the frame-wide DoG ladder's block energy tables. Declared here,
// ahead of its own section further down, because dt_iop_contrast_gui_data_t
// needs the type; see that section for what builds and frees one.
typedef struct _ct_ladder_t
{
  int    nrungs;
  double sigma[CT_MAX_BANDS];   // §3.1: rung's own lower-boundary sigma, level-0 pixels
  double lambda[CT_MAX_BANDS];  // band-centre wavelength (sigma * CT_SIGMA_TO_LAMBDA), level-0 pixels
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
  float feathering[CT_BANDS];   // §3.3: per-band edge protection, same order as sigma/gain
  float feathering_base;        // from edge_protection/filter_iterations, before the per-band scaling
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
  double ladder_sigma[CT_MAX_BANDS];   // §3.1: rung's own lower-boundary sigma, level-0 px
  double ladder_lambda[CT_MAX_BANDS];
  double ladder_step[CT_MAX_BANDS];
  double ladder_noise_floor[CT_MAX_BANDS];  // §2.4, frame-wide, published the same way
  dt_iop_roi_t ladder_roi_in;  // the roi_in the ladder above was built from

  // §3.1: per-band block SAT tables for the module's own delivered bands,
  // built alongside the ladder in the same guarded preview pass (see
  // _decompose_and_accumulate's optional _ct_band_tables_t argument) and
  // queried the same way to calibrate the picker's linear H_k model against
  // what eigf's edge-awareness actually delivers (research.md §5.8).
  // components is fixed at 2*CT_BANDS -- unlike the ladder's nrungs, CT_BANDS
  // never changes, so this pd needs no resize-on-change dance.
  dt_preview_data_t band_pd;
  int band_nbands;               // how many of CT_BANDS survived the pass that built band_pd
  float band_sigma[CT_BANDS];    // finest-first, pixels of that same pass's roi (== ladder_roi_in)

  // §3.2: the last successful pick's own raw per-rung spectrum and fit, for
  // the graph's live overlay -- a record of the last measurement, drawn
  // every _area_draw regardless of whether the picker itself is still
  // "fresh". The fit's scalar fields are stored individually rather than as
  // a _ct_fit_t so this struct doesn't need that type's (later) definition.
  gboolean spectrum_valid;
  int spectrum_nrungs;
  double spectrum_lambda[CT_MAX_BANDS];
  double spectrum_energy[CT_MAX_BANDS];
  double spectrum_noise, spectrum_self_similar, spectrum_texture, spectrum_tau, spectrum_beta;
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
// clear sky sits well below it; the grain of dry asphalt, about the faintest
// thing anyone would pick deliberately, sits comfortably above (~40x margin,
// implementation-plan-2.md §7.1, re-verified against Phase 1-5's ladder on 19
// real crops from a mixed image set -- unchanged). Out-of-focus background is
// *not* reliably below it and that is correct, not a bug: a busy bokeh patch
// (blown highlights, dark blobs) can carry more real broadband energy than a
// deliberately-picked faint texture, so forcing it under this floor would
// also have to swallow real texture and destroy the separation the floor
// exists for.
#define CT_FLAT_RMS_EV 0.008
#define CT_FLAT_ENERGY (CT_FLAT_RMS_EV * CT_FLAT_RMS_EV)
// grid resolution of the model fit's texture size, per octave
#define CT_FIT_STEPS_PER_OCTAVE 8.0
// the fit's other grid: the self-similar spectrum's slope. implementation-
// plan-2.md §7.3: research.md §5.3's figure was for *linear* radiance and
// this ladder measures log2 luminance, whose slope need not match -- the old
// [1.6, 3.0] range pegged 8 of 18 real fits (44%) at one of its own bounds,
// on a 19-crop mixed real-image set. Widened and re-centred from where an
// unconstrained grid search (deliberately over-wide, [0.2, 6.0]) actually
// settled real texture content: clean picks (petals, fur, skin, asphalt,
// foliage, architecture) cluster 1.8-3.4 with no pegging under this range;
// only genuinely flat/ambiguous content (clear sky, an out-of-focus patch --
// exactly what CT_FLAT_ENERGY above is supposed to catch first) still rails
// against the top, which is a property of that content having no real
// texture to fit, not of the range being too narrow for it.
#define CT_FIT_BETA_MIN 1.4
#define CT_FIT_BETA_MAX 4.0
#define CT_FIT_BETA_STEPS 12  // 13 values, CT_FIT_BETA_MIN .. CT_FIT_BETA_MAX
// rungs this far below the peak are noise, and in log space they would
// otherwise dominate the residual
#define CT_ENERGY_FLOOR 1e-6
// how far an idealized hump is expected to sit from a real texture's ladder,
// as a fraction. it is the floor under every rung's error bar: past a couple
// of hundred independent samples a rung stops getting more trustworthy, so
// extra pixels stop buying it extra weight. implementation-plan-2.md §7.4:
// unchanged, re-verified -- across the same 19-crop set, weighting by
// n_indep (§1.2) instead of a pixel count never produced a fit dominated by
// the fine rungs (no pathological texture/self_similar split, no fit that
// ignored the coarse end); this floor is doing its job against the wider
// weight spread §1.2 introduced.
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

// implementation-plan-2.md §6.1: why a pick was refused, so the caller can
// say something specific instead of one message covering "too small" and
// "nothing here" alike.
typedef enum _ct_fit_refusal_t
{
  CT_FIT_REFUSED_SPAN,  // fewer than CT_MIN_BANDS rungs, or less than CT_MIN_SPAN octaves
  CT_FIT_REFUSED_FLAT   // nothing above CT_FLAT_ENERGY at any surviving rung
} _ct_fit_refusal_t;

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
// floor anywhere in the box -- *reason says which (implementation-plan-2.md
// §6.1), so the caller can say something specific.
static gboolean _fit_spectrum(const double *const restrict sigma,
                              const double *const restrict energy,
                              const double *const restrict weight,
                              const int n,
                              const double noise_prior,
                              _ct_fit_t *const restrict fit,
                              _ct_fit_refusal_t *const restrict reason)
{
  // the smallest area that can be measured at all covers exactly one octave,
  // so this comparison is met on the nose there and is given a rounding's
  // worth of slack rather than being left to turn on an ulp. sigma and
  // lambda differ by the fixed CT_SIGMA_TO_LAMBDA factor, so the ratio this
  // compares is the same either way.
  if(n < CT_MIN_BANDS || sigma[n - 1] < CT_MIN_SPAN * sigma[0] * (1.0 - 1e-9))
  {
    *reason = CT_FIT_REFUSED_SPAN;
    return FALSE;
  }

  double peak_e = 0.0;
  for(int i = 0; i < n; i++) peak_e = fmax(peak_e, energy[i]);
  // nothing there at any scale: a blank sky, a blown highlight, a black frame
  if(peak_e <= CT_FLAT_ENERGY)
  {
    *reason = CT_FIT_REFUSED_FLAT;
    return FALSE;
  }

  double s[CT_MAX_BANDS];  // s = sigma^2, the model's own scale variable
  for(int i = 0; i < n; i++) s[i] = sigma[i] * sigma[i];

  const gboolean fix_noise = noise_prior >= 0.0;
  const gboolean init_active[3] = { !fix_noise, TRUE, TRUE };

  // candidate texture sizes: from half the finest rung to the coarsest one,
  // same range `_fit_texture_scale` scanned and for the same reason -- the
  // top of it puts the hump's peak just past the end of the ladder, as far
  // as the rising flank alone can honestly be pushed.
  const double lo = sigma[0] * 0.5;
  const double hi = sigma[n - 1];
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

// §2.5/research.md §5.6: S(sigma) = A*G(sigma;tau) + C*sigma^(beta-2) (real
// scene detail) and N(sigma) = N*G(sigma;0) (noise), evaluated in exactly the
// (s = sigma^2) basis _fit_spectrum solved in -- reusing _dog_shape as G is
// what keeps this consistent with the fit rather than introducing a second,
// unrelated normalisation. implementation-plan-2.md §3.2: sigma in, not
// lambda -- every caller now converts at its own call site (§3.3's table),
// not here.
static inline void _ct_fit_eval(const _ct_fit_t *const fit, const double sigma,
                                double *const S, double *const N)
{
  const double s = sigma * sigma;
  *S = fit->texture * _dog_shape(s, fit->tau)
     + fit->self_similar * pow(s, (fit->beta - 2.0) * 0.5);
  *N = fit->noise * _dog_shape(s, 0.0);
}

// §2.5: which of research.md §5.6's target shapes a fit earns. TEXTURE is
// the default; DETAIL is the A-negligible fallback -- "a self-similar area
// with no size", `_fit_curve_from_box` below decides which.
// implementation-plan-2.md §8.1: CT_TARGET_EQUALIZE is research.md §5.6's
// third shape, "boost what's weak" rather than TEXTURE's "boost whatever
// carries the most energy" -- already shipped, unchanged, as the "flatten
// spectrum" preset; §8.1 only wires it into the live picker path so a pick
// can use it too. Not yet chosen *between* TEXTURE and EQUALIZE for the
// picker -- that is §8.2, decided separately with rendered crops, not code.
typedef enum _ct_target_mode_t
{
  CT_TARGET_TEXTURE  = 0,
  CT_TARGET_DETAIL   = 1,
  CT_TARGET_EQUALIZE = 2
} _ct_target_mode_t;

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

// implementation-plan-2.md §7.2: unchanged, re-verified against Phase 2's
// ladder on real content, including two deliberately high-ISO/deep-shadow
// crops (where sensor noise should be most visible if this were going to
// break) -- every rung above octave 2 (index >= 3*CT_SCALES_PER_OCTAVE)
// still comes back -1 (no near-Gaussian block found), and
// _ladder_estimate_noise's min-over-rungs still lands on one of the fine
// rungs, giving a small, sane N rather than swamping the fit (§7.4 confirms
// the fits themselves stayed sane downstream of it).
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
static double _ladder_estimate_noise(const double *const restrict sigma,
                                     const double *const restrict noise_floor,
                                     const int nrungs)
{
  double best = -1.0;
  for(int r = 0; r < nrungs; r++)
  {
    if(noise_floor[r] < 0.0) continue;
    const double g0 = _dog_shape(sigma[r] * sigma[r], 0.0);
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
  const float minv = -1.0e6f, maxv = 1.0e6f;

  // §2: bring octave 0's base up to absolute CT_SIGMA_BASE once, so every
  // octave starts from the same absolute sigma in its own pixel units.
  // Without this, rung[0] of octave 0 is the raw (unblurred) base while
  // every later octave's base already carries sigma = CT_SIGMA_BASE from the
  // previous octave's own rung[CT_SCALES_PER_OCTAVE] (subsampled by two, so
  // still CT_SIGMA_BASE in the new grid's own units) -- octave 0 alone would
  // then measure every band a sqrt(1 + (CT_SIGMA_BASE/sigma_s)^2) too wide.
  // implementation-plan-2.md §2.
  {
    dt_gaussian_t *const gs =
      dt_gaussian_init((int)width, (int)height, 1, &maxv, &minv, CT_SIGMA_BASE,
                       DT_IOP_GAUSSIAN_ZERO);
    if(!gs) ok = FALSE;
    else
    {
      dt_gaussian_blur(gs, level, next);
      dt_gaussian_free(gs);
      memcpy(level, next, npixels * sizeof(float));
    }
  }

  for(int octave = 0;
      ok && octave < CT_MAX_OCTAVES && nrungs + CT_SCALES_PER_OCTAVE <= CT_MAX_BANDS;
      octave++)
  {
    // rung[0] **is** the base (already at absolute CT_SIGMA_BASE); every
    // later rung is an incremental blur from it, not a fresh blur of the
    // base from sigma 0 -- that incremental step is what keeps every
    // octave's rungs at the same absolute sigma the labels claim.
    memcpy(rung[0], level, cw * ch * sizeof(float));
    for(int s = 1; s <= CT_SCALES_PER_OCTAVE; s++)
    {
      const float target = CT_SIGMA_BASE * exp2f((float)s / CT_SCALES_PER_OCTAVE);
      const float inc = sqrtf(target * target - CT_SIGMA_BASE * CT_SIGMA_BASE);
      dt_gaussian_t *const g =
        dt_gaussian_init((int)cw, (int)ch, 1, &maxv, &minv, inc, DT_IOP_GAUSSIAN_ZERO);
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

      // implementation-plan-2.md §3.1: label the rung by its own lower-
      // boundary sigma, not the geometric mean of its two rung sigmas -- the
      // DoG's actual peak sits at CT_SIGMA_TO_LAMBDA * sigma_lower (the exact
      // formula, not "within a percent"), which is 1.41x finer than the old
      // geometric-mean label claimed.
      const double sigma_s = CT_SIGMA_BASE * exp2((double)s / CT_SCALES_PER_OCTAVE);
      ladder->sigma[nrungs] = sigma_s * step;
      ladder->lambda[nrungs] = ladder->sigma[nrungs] * CT_SIGMA_TO_LAMBDA;
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

// §3.1: per-band block S1/S2 tables for the module's own delivered bands,
// built over the same CT_BLOCK grid the DoG ladder (§2.1) uses so a box query
// is the same 4-lookups-per-band shape -- but always at step = 1 (module
// bands are full resolution by the time they reach here, direct or upsampled
// pyramid alike). scratch_* are owned by the caller and reused across every
// band k in turn; sat1/sat2 hold CT_BANDS separate (bw+1)x(bh+1) tables,
// band-major, one built per k as _decompose_and_accumulate/
// _accumulate_pyramid_bands finish computing that band's b_k.
typedef struct _ct_band_tables_t
{
  size_t bw, bh;
  double *sat2, *sat1;      // CT_BANDS * (bw+1) * (bh+1) doubles each
  float *scratch_full;      // npixels, this band's own b_k
  double *scratch_blk2, *scratch_blk1;  // bw*bh, this band's own blocks
} _ct_band_tables_t;

// accumulate one band's already-computed b_k (bt->scratch_full) into its own
// slot of bt's SAT tables. Reuses §2.1's block/SAT helpers verbatim -- they
// were already generic over "one band's array + its own step", and a module
// band's step is simply 1.
static void _band_tables_accumulate(_ct_band_tables_t *const restrict bt,
                                    const int k, const size_t width, const size_t height)
{
  _ladder_accumulate_blocks(bt->scratch_full, width, height, 1.0, bt->bw, bt->bh,
                            bt->scratch_blk2, bt->scratch_blk1);
  const size_t sat_stride = (bt->bw + 1) * (bt->bh + 1);
  _ladder_build_sat(bt->scratch_blk2, bt->bw, bt->bh, bt->sat2 + (size_t)k * sat_stride);
  _ladder_build_sat(bt->scratch_blk1, bt->bw, bt->bh, bt->sat1 + (size_t)k * sat_stride);
}

// §3.1: dt_preview_data_fill_t for _ct_band_tables_t, the same reshape
// _ladder_fill_cb does for the ladder but with CT_BANDS fixed instead of a
// variable nrungs.
static void _band_fill_cb(void *const user_data, float *const buf, const size_t nelems)
{
  const _ct_band_tables_t *const bt = (const _ct_band_tables_t *)user_data;
  const size_t sw = bt->bw + 1, sh = bt->bh + 1;
  const size_t comps = (size_t)(2 * CT_BANDS);
  (void)nelems;  // == sw * sh * comps, by construction of the caller's resize

  for(size_t y = 0; y < sh; y++)
    for(size_t x = 0; x < sw; x++)
    {
      float *const dst = buf + (y * sw + x) * comps;
      for(int k = 0; k < CT_BANDS; k++)
      {
        dst[2 * k]     = (float)bt->sat2[(size_t)k * sw * sh + y * sw + x];
        dst[2 * k + 1] = (float)bt->sat1[(size_t)k * sw * sh + y * sw + x];
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
// prev_log is the running previous band's log2(blur) (research.md §2.1's
// L_{k-1}), full res, seeded by the caller from the last direct band and
// updated in place here as each pyramid level is folded in -- see the
// correctness note above _decompose_and_accumulate for why this has to be
// incremental rather than always diffed against the original luminance.
__DT_CLONE_TARGETS__
static void _accumulate_pyramid_bands(const float *const restrict lum,
                                      float *const restrict prev_log,
                                      float *const restrict correction,
                                      float *const restrict coarsest,
                                      float *const restrict full_scratch,
                                      const size_t width, const size_t height,
                                      const dt_iop_contrast_data_t *const d,
                                      const int direct_bands,
                                      const int display_band,
                                      _ct_band_tables_t *const restrict bt)  // §3.1, NULL to skip
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
    fast_eigf_surface_blur(level_blur, base_w, base_h, local_sigma, d->feathering[k], d->iterations,
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
      const float log_blur = log2f(fmaxf(full_scratch[p], NORM_MIN));
      const float b_k = prev_log[p] - log_blur;
      if(is_display) correction[p] = b_k;
      else if(detail_mode) correction[p] += b_k;
      else if(display_band < 0) correction[p] += gain_minus_one * b_k;
      if(bt) bt->scratch_full[p] = b_k;
      prev_log[p] = log_blur;
    }
    if(bt) _band_tables_accumulate(bt, k, width, height);

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
// b_k is research.md §2.1's incremental band: log2(blur_{k-1}) - log2(blur_k),
// diffed against the *previous* band's own blur (blur_{-1} = the untouched
// luminance), not always against the original -- that is what makes
// sum_k b_k telescope to a single log2(L) - log2(blur_{nbands-1}) highpass
// when every gain is equal, which is the property legacy_params's v1
// conversion and DT_CT_MASK_DETAIL both rely on to reproduce v1's own
// single-band behaviour exactly. A cumulative log2(L) - log2(blur_k) here
// (diffing every band against the original image) does not telescope and
// silently over-boosts whenever more than one band is open at once.
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
                                      const int display_band,
                                      _ct_band_tables_t *const restrict bt)  // §3.1, NULL to skip
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
    fast_eigf_surface_blur(blur, width, height, d->sigma[k], d->feathering[k], d->iterations,
                           DT_GF_BLENDING_LINEAR, 1.0f,
                           0.0f, NORM_MIN, 4.0f);

    const float gain_minus_one = d->gain[k] - 1.0f;
    const gboolean is_display = (display_band == k);
    const gboolean detail_mode = (display_band == -2);

    DT_OMP_FOR()
    for(size_t p = 0; p < npixels; p++)
    {
      const float log_blur = log2f(fmaxf(blur[p], NORM_MIN));
      const float b_k = log_lum[p] - log_blur;  // log_lum here holds band (k-1)'s own blur, not the original
      if(is_display) correction[p] = b_k;
      else if(detail_mode) correction[p] += b_k;
      else if(display_band < 0) correction[p] += gain_minus_one * b_k;
      if(bt) bt->scratch_full[p] = b_k;
      log_lum[p] = log_blur;  // becomes band (k+1)'s "previous"
    }
    if(bt) _band_tables_accumulate(bt, k, width, height);

    if(k == d->nbands - 1) memcpy(coarsest, blur, npixels * sizeof(float));
  }

  // log_lum now holds the last direct band's own log2(blur) -- exactly the
  // "previous" state _accumulate_pyramid_bands needs to keep the incremental
  // chain going into the coarse tail, per the correctness note above.
  if(direct_bands < d->nbands)
    _accumulate_pyramid_bands(lum, log_lum, correction, coarsest, blur, width, height,
                              d, direct_bands, display_band, bt);

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

  // §2.2/§3.1: keep the frame-wide DoG ladder and the module's own per-band
  // block tables current while the module is expanded, on the untiled
  // preview pipe -- a tile is not a whole frame. color_picker_apply()
  // queries both synchronously on the GUI thread; see there for the box
  // query and the fit/calibration they feed.
  const gboolean update_calibration_data =
    g && self->dev->gui_attached && self->expanded
    && dt_pipe_is_preview(piece->pipe) && !piece->pipe->tiling;

  _ct_band_tables_t band_tables = { 0 };
  gboolean have_band_tables = FALSE;
  if(update_calibration_data)
  {
    band_tables.bw = (width + CT_BLOCK - 1) / CT_BLOCK;
    band_tables.bh = (height + CT_BLOCK - 1) / CT_BLOCK;
    const size_t bstride = (band_tables.bw + 1) * (band_tables.bh + 1);
    band_tables.sat2 = dt_alloc_align_double(bstride * CT_BANDS);
    band_tables.sat1 = dt_alloc_align_double(bstride * CT_BANDS);
    band_tables.scratch_full = dt_alloc_align_float(npixels);
    band_tables.scratch_blk2 = dt_alloc_align_double(band_tables.bw * band_tables.bh);
    band_tables.scratch_blk1 = dt_alloc_align_double(band_tables.bw * band_tables.bh);
    have_band_tables = band_tables.sat2 && band_tables.sat1 && band_tables.scratch_full
                      && band_tables.scratch_blk2 && band_tables.scratch_blk1;
  }

  if(update_calibration_data)
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
      memcpy(g->ladder_sigma, built.sigma, sizeof(g->ladder_sigma));
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

  _decompose_and_accumulate(luminance, correction, coarsest, width, height, d, display_band,
                            have_band_tables ? &band_tables : NULL);

  // §3.1: publish the per-band tables _decompose_and_accumulate just filled,
  // the same way §2.2 publishes the ladder -- band_sigma/band_nbands are
  // this pass's own d->sigma/d->nbands, kept alongside so a later query
  // evaluates the calibration model at the sigmas that actually produced
  // these energies rather than whatever the picker's current params say.
  if(have_band_tables)
  {
    dt_preview_data_store(&g->band_pd, band_tables.bw + 1, band_tables.bh + 1, piece,
                          _band_fill_cb, &band_tables);

    dt_iop_gui_enter_critical_section(self);
    g->band_nbands = d->nbands;
    memset(g->band_sigma, 0, sizeof(g->band_sigma));
    memcpy(g->band_sigma, d->sigma, sizeof(float) * MIN(d->nbands, CT_BANDS));
    dt_iop_gui_leave_critical_section(self);
  }
  dt_free_align(band_tables.sat2);
  dt_free_align(band_tables.sat1);
  dt_free_align(band_tables.scratch_full);
  dt_free_align(band_tables.scratch_blk2);
  dt_free_align(band_tables.scratch_blk1);

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
  float sigma_ref = 1.0f;  // §3.3: the finest surviving band at this scale, set below
  for(int k = CT_BANDS - 1; k >= 0; k--)
  {
    const float D = CT_BAND_D0 + k + 0.5f + d->scale_shift;
    const float diameter = exp2f(-D) * S * roi_in->scale;
    const float sigma = 0.5f * (diameter - 1.0f);

    if(nbands == 0 && sigma < 0.7f) continue;  // unresolvable fine tail: drop
    if(nbands == 0) sigma_ref = fmaxf(sigma, 0.7f);

    d->sigma[nbands] = sigma;
    d->gain[nbands] = d->band[k];
    // §3.3/research.md §2.4: eigf's a = v/(v+eps) saturates toward a = 1 (no
    // blurring at all) as the window grows, so a single global eps leaves
    // coarse bands empty on most ordinary photographs -- phase0-band-
    // energy.md found the coarsest 1-4 of 9 bands going exactly to zero on
    // three of four test images (portrait, sunset, flower), and recommended
    // promoting this from a Phase-3 "only if" to required. Scale eps up
    // with the band's own sigma relative to the finest surviving one, so
    // coarse bands keep real edge-tolerance instead of saturating away.
    d->feathering[nbands] =
      d->feathering_base * powf(fmaxf(sigma, 0.7f) / sigma_ref, CT_FEATHERING_EXPONENT);
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
  // §3.3: this is now the *base* value modify_roi_in scales per band
  // (d->feathering[] is what process() actually reads); modify_roi_in runs
  // after commit_params for the same pipe pass and needs d->band[] (also
  // written here) before it can compute a per-band sigma to scale by.
  d->feathering_base = default_feathering * powf(2.0f, -p->edge_protection) / (p->filter_iterations * p->filter_iterations);
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

// ---------------------------------------------------------------------------
// §2.5: from the fit to a target gain curve, and from that curve to nodes
// ---------------------------------------------------------------------------
//
// research.md §5.6: the picker's job stops at *shape*, never strength -- the
// caller (color_picker_apply) is what turns this into an actual gain curve,
// by lerping between 1 (no effect) and the master gain along the shape
// below, so dragging the master gain afterwards keeps behaving predictably.

// §2.5/research.md §5.6: shape(sigma), peak-normalised where the table says
// so -- T_hat carries its own peak-1 normalisation (over the grid actually
// evaluated, since that is the only "max(A*G)" available here); DETAIL's
// S/(S+N) is used exactly as it falls out, per the table, with no further
// rescaling.
//
// implementation-plan-2.md §8.1: CT_TARGET_EQUALIZE is different in *kind*,
// not degree, from the other two -- research.md §5.6's own table gives it as
// a bounded absolute curve, clamp((E_ref/S_hat)^alpha) * S/(S+N), already
// shipped unchanged as the "flatten spectrum" preset's own expression (§3.4)
// -- not a [0,1] boost shape with a separate strength knob. shape[] here for
// CT_TARGET_EQUALIZE *is* that curve directly: the caller must use it as the
// target curve as-is, not lerp it between 1 and a master gain the way
// TEXTURE/DETAIL's shape[] is. E_ref is evaluated at the same geometric-mean-
// of-the-grid reference point the preset uses, against this box's own fit
// rather than the preset's synthetic self-similar assumption.
#define CT_EQUALIZE_ALPHA 0.4
#define CT_EQUALIZE_GAIN_LO 0.3
#define CT_EQUALIZE_GAIN_HI 2.5

static void _target_curve(const _ct_fit_t *const fit, const _ct_target_mode_t mode,
                          const double *const restrict sigma_grid, const int m,
                          double *const restrict shape)
{
  double peak_tex = 0.0;
  if(mode == CT_TARGET_TEXTURE)
    for(int j = 0; j < m; j++)
      peak_tex = fmax(peak_tex, fit->texture * _dog_shape(sigma_grid[j] * sigma_grid[j], fit->tau));

  double s_ref = 0.0, n_ref = 0.0;
  if(mode == CT_TARGET_EQUALIZE && m > 0)
  {
    const double sigma_mid = sqrt(sigma_grid[0] * sigma_grid[m - 1]);
    _ct_fit_eval(fit, sigma_mid, &s_ref, &n_ref);
  }

  for(int j = 0; j < m; j++)
  {
    double S, N;
    _ct_fit_eval(fit, sigma_grid[j], &S, &N);
    const double wiener = S / fmax(S + N, DBL_MIN);

    if(mode == CT_TARGET_TEXTURE)
    {
      const double that = peak_tex > 0.0
        ? (fit->texture * _dog_shape(sigma_grid[j] * sigma_grid[j], fit->tau)) / peak_tex : 0.0;
      shape[j] = that * wiener;
    }
    else if(mode == CT_TARGET_EQUALIZE)
    {
      const double eq = pow(s_ref / fmax(S, DBL_MIN), CT_EQUALIZE_ALPHA) * wiener;
      shape[j] = CLAMP(eq, CT_EQUALIZE_GAIN_LO, CT_EQUALIZE_GAIN_HI);
    }
    else
    {
      shape[j] = wiener;
    }
  }
}

// §2.5/research.md §5.7: least-squares projection of a target curve onto the
// module's own band gains. Point-sampling g_target at each node's own
// wavelength would be wrong -- the bands overlap too heavily (§2.2) -- so
// this solves for the {g_k} whose H_k basis best reproduces the whole curve
// instead, with a second-difference penalty for smoothness (the fitted model
// is already the regulariser against per-rung noise; this is what keeps nine
// mostly-collinear H_k columns from chasing that noise into an oscillating
// curve).
//
// H_k(lambda) = HP_k(lambda) - HP_{k-1}(lambda), HP_k(lambda) = 1 -
// exp(-2*pi^2*sigma_k^2/lambda^2) is the cumulative fraction of energy at
// wavelength `lambda` band k's own highpass (relative to its boundary sigma)
// would capture; the incremental H_k is what setting gain_k alone adds to
// the module's net response, treating boundary -1 as sigma = 0 (nothing
// captured before band 0).
//
// sigma[] and gains[] both run finest-first (sigma increasing with k), which
// is d->sigma[]'s own convention (§1.2) -- the caller maps back to param
// (coarsest-first) order.
#define CT_PROJECT_GRID 100
// second-difference penalty weight (relative to one grid point's own unit
// weight, scaled by grid size below) -- small enough not to flatten a real
// single-octave hump, large enough that nine mostly-collinear H_k columns
// don't chase per-point noise into an oscillating curve. no closed-form
// value here; picked by eye against a synthetic single-hump target and left
// generous rather than tight, since implementation-plan.md §2.5's own
// acceptance leans on the fit -- not this regulariser -- to keep the curve
// honest.
#define CT_PROJECT_SMOOTHNESS 0.05

// calibration is §3.1's per-band E_module,k/E_predicted,k ratio (NULL, or any
// entry at 1.0, means uncalibrated) -- one scalar multiplier on band k's own
// H_k column, so the solve asks for proportionally more (gain_k - 1) wherever
// eigf's edge-awareness is known to deliver less than this linear model
// would (research.md §5.8). Rows are unaffected: the smoothness penalty acts
// on the {g_k} output directly and has no H_k of its own to calibrate.
//
// implementation-plan-2.md §4.4: gain_lo/gain_hi are the envelope the
// caller's own target curve promised -- research.md §5.6 and _target_curve's
// own comment say the picker sets shape, never strength, so no band should
// come back outside it. The second-difference penalty above can ring past
// the envelope on a two-to-three-band hump (0.974/0.81 measured on real
// picks, both clamp violations, neither a real measurement); this is a hard
// clamp rather than a clamp-and-resolve active set because that ringing was
// small in practice -- revisit if a legitimate curve is measured to flatten
// visibly against it. The picker passes [min(1,master), max(1,master)];
// presets pass their own envelope directly since some (soften) are
// deliberately < 1 and one (flatten spectrum) straddles 1 on both sides,
// neither of which a single "master" scalar can express.
static gboolean _project_to_bands(const double *const restrict lambda_grid,
                                  const double *const restrict g_target,
                                  const int m,
                                  const float *const restrict sigma,
                                  const int nbands,
                                  const float *const restrict calibration,
                                  const float gain_lo, const float gain_hi,
                                  float *const restrict gains)
{
  if(nbands < 2 || m < 2) return FALSE;

  const int nreg = nbands - 2;
  const size_t rows = (size_t)m + (size_t)MAX(nreg, 0);

  float *const restrict A = dt_alloc_align_float(rows * (size_t)nbands);
  float *const restrict y = dt_alloc_align_float(rows);
  if(!A || !y) { dt_free_align(A); dt_free_align(y); return FALSE; }

  memset(A, 0, rows * (size_t)nbands * sizeof(float));

  for(int j = 0; j < m; j++)
  {
    const double lambda = lambda_grid[j];
    for(int k = 0; k < nbands; k++)
    {
      const double sigma_km1 = (k == 0) ? 0.0 : (double)sigma[k - 1];
      const double sigma_k = (double)sigma[k];
      const double hp_km1 = 1.0 - exp(-2.0 * M_PI * M_PI * sigma_km1 * sigma_km1 / (lambda * lambda));
      const double hp_k   = 1.0 - exp(-2.0 * M_PI * M_PI * sigma_k   * sigma_k   / (lambda * lambda));
      const float r = calibration ? calibration[k] : 1.0f;
      A[j * nbands + k] = (float)(hp_k - hp_km1) * r;
    }
    y[j] = (float)(g_target[j] - 1.0);
  }

  const double w = sqrt(CT_PROJECT_SMOOTHNESS * (double)m);
  for(int r = 0; r < nreg; r++)
  {
    const int row = m + r;
    A[row * nbands + r]     = (float)w;
    A[row * nbands + r + 1] = (float)(-2.0 * w);
    A[row * nbands + r + 2] = (float)w;
    y[row] = 0.0f;
  }

  const gboolean ok = pseudo_solve(A, y, rows, (size_t)nbands, FALSE);
  if(ok)
    for(int k = 0; k < nbands; k++)
      gains[k] = CLAMP(y[k] + 1.0f, gain_lo, gain_hi);

  dt_free_align(A);
  dt_free_align(y);
  return ok;
}

// ---------------------------------------------------------------------------
// §3.1: per-band calibration -- research.md §5.8
// ---------------------------------------------------------------------------
//
// The DoG ladder is linear; eigf is not (§5.8's table: as little as 9% of a
// hard edge's local deviation reaches the high-pass at the default edge
// protection). A curve fitted from the linear ladder therefore over-promises
// wherever the picked box has real local contrast, unless the projection
// above is told how much of its own H_k a band actually delivers.
//
// research.md §2.2's peak wavelength of H_k = HP_k - HP_{k-1}, for an
// octave-spaced ladder; the finest band (sigma_km1 = 0) is a shelf, not a
// bump (same section), so its own half-amplitude wavelength stands in.
static double _band_peak_lambda(const double sigma_km1, const double sigma_k)
{
  if(sigma_km1 <= 0.0) return 2.0 * M_PI * sigma_k / sqrt(2.0 * log(2.0));
  const double num = 2.0 * (sigma_k * sigma_k - sigma_km1 * sigma_km1);
  const double den = log((sigma_k * sigma_k) / (sigma_km1 * sigma_km1));
  return M_PI * sqrt(num / den);
}

// implementation-plan-2.md §5.1: peak wavelength of an octave-spaced H_k
// pair, i.e. _band_peak_lambda specialised to sigma_k = 2*sigma_km1 --
// pi*sqrt(6/ln 4). The finest band (sigma_km1 = 0) is a shelf, not a bump
// (_band_peak_lambda's own comment), so it will not land exactly on its
// node under the formula below; that is correct and should be left alone.
#define CT_BAND_PEAK_FACTOR 6.5357852

// map a wavelength (in some roi's own pixels, or a frame-relative fraction
// of the long edge if roi_long_edge is 1.0) to a graph x fraction. Node k is
// drawn at (k+0.5)/CT_BANDS (_graph_curve_from_params); band k's own H_k
// actually peaks at CT_BAND_PEAK_FACTOR * sigma_lower, sigma_lower being
// the next-finer band's own boundary sigma (half of band k's own, under
// §4.1's octave-spaced frame-relative ladder) -- working through §4.1's
// sigma[k] = 2^-(D0+k+1.5) puts that peak at
// 2^-(D0+k+0.5) * (CT_BAND_PEAK_FACTOR/4). Anchoring the axis there,
// instead of at the nominal detail level the old formula used, is what
// makes a rung/preset shape and the node whose H_k actually responds to it
// land at the same x -- the old nominal-level axis drew the spectrum
// 0.7083606 octave toward the coarse end of the band it belonged to
// (implementation-plan-2.md §5.1's own measurement). Verified here as an
// identity against _band_peak_lambda for all nine bands, per §5.1's own
// acceptance criterion; the doc's own inline code snippet has this
// correction term's sign backwards (confirmed by that check -- an additive
// +log2(F/4), not the doc's -log2(F/4)). Ignores scale_shift, exactly as
// the nodes' own fixed screen positions do, so a rung/preset shape and the
// node it nominally corresponds to line up regardless of where scale_shift
// has since moved the *physical* meaning of that node. Shared by §3.2's
// graph overlay and §3.4's analytic preset shapes.
static float _spectrum_lambda_to_x(const double lambda, const double roi_long_edge)
{
  const double d = -log2(lambda / fmax(roi_long_edge, 1.0)) + log2(CT_BAND_PEAK_FACTOR / 4.0);
  return CLAMP((float)((d - CT_BAND_D0) / (double)CT_BANDS), 0.0f, 1.0f);
}

// query §3.1's per-band block tables (built alongside the ladder in the same
// guarded preview pass -- see process()'s comment above the ladder build)
// for the box the picker used. e_module/sigma_out are filled finest-first,
// for the first *nbands_out entries only -- sigma_out is band_sigma as of the
// pass that produced e_module, not the picker's possibly-since-changed
// current node placement, so the two stay internally consistent with each
// other even if they drift a little from "right now".
//
// Returns FALSE (leaving every output untouched) if there is nothing fresh
// to offer -- module just opened, or a param change raced the preview pipe.
// Calibration is a refinement on an already-working picker, not a
// precondition for one (research.md §5.10's framing for the analogous
// second-picker question) -- the caller falls back to uncalibrated rather
// than stalling or refusing the pick over a missing table.
static gboolean _query_band_energy(dt_iop_module_t *self, const int *const box,
                                   double *const restrict e_module,
                                   double *const restrict sigma_out,
                                   int *const restrict nbands_out)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;
  if(!dt_preview_data_is_fresh(&g->band_pd)) return FALSE;

  dt_iop_gui_enter_critical_section(self);

  const size_t sat_w = g->band_pd.width, sat_h = g->band_pd.height;
  const gboolean have_data =
    g->band_pd.buf && sat_w > 1 && sat_h > 1
    && g->band_pd.components == (size_t)(2 * CT_BANDS) && g->band_nbands > 0;

  if(have_data)
  {
    const size_t bw = sat_w - 1, bh = sat_h - 1;
    const size_t bx0 = MIN(bw, (size_t)MAX(box[0], 0) / CT_BLOCK);
    size_t bx1 = MIN(bw, (size_t)(MAX(box[2], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(bx1 <= bx0) bx1 = MIN(bw, bx0 + 1);
    const size_t by0 = MIN(bh, (size_t)MAX(box[1], 0) / CT_BLOCK);
    size_t by1 = MIN(bh, (size_t)(MAX(box[3], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(by1 <= by0) by1 = MIN(bh, by0 + 1);
    // module bands are always full res (direct, or upsampled pyramid alike)
    // -- step = 1, unlike the ladder's per-rung decimation.
    const double n_eff =
      fmax((double)(bx1 - bx0) * (double)(by1 - by0) * (double)(CT_BLOCK * CT_BLOCK), 1.0);

    const int nbands = MIN(g->band_nbands, CT_BANDS);
    const size_t comps = (size_t)(2 * CT_BANDS);
    const float *const restrict buf = g->band_pd.buf;
    for(int k = 0; k < nbands; k++)
    {
      const double s2 = buf[(by1 * sat_w + bx1) * comps + 2 * k]
                       - buf[(by0 * sat_w + bx1) * comps + 2 * k]
                       - buf[(by1 * sat_w + bx0) * comps + 2 * k]
                       + buf[(by0 * sat_w + bx0) * comps + 2 * k];
      e_module[k] = s2 / n_eff;
      sigma_out[k] = (double)g->band_sigma[k];
    }
    *nbands_out = nbands;
  }

  dt_iop_gui_leave_critical_section(self);
  return have_data;
}

// r_k = E_module,k / E_predicted,k, clamped against a near-empty band's
// E_predicted blowing the ratio up rather than trusted at face value --
// this is an empirical correction, not a physical law, and both the box and
// the fit are noisy. Physically eigf never delivers *more* than the linear
// model predicts (1 - a = eps/(v+eps) <= 1 always), but the fit's own beta
// need not exactly match the module's own bands, so a little headroom above
// 1 is left rather than hard-clamped there. calibration defaults every band
// to 1 (uncalibrated) first, so a stale or missing table just skips the
// refinement instead of failing the pick.
#define CT_CALIBRATION_MIN 0.05
#define CT_CALIBRATION_MAX 3.0
#define CT_CALIBRATION_FLOOR 1e-9

static void _compute_band_calibration(dt_iop_module_t *self, const int *const box,
                                      const float long_edge,
                                      const _ct_fit_t *const fit,
                                      float *const restrict calibration)
{
  for(int k = 0; k < CT_BANDS; k++) calibration[k] = 1.0f;

  double e_module[CT_BANDS], sigma_d[CT_BANDS];
  int nbands = 0;
  if(!_query_band_energy(self, box, e_module, sigma_d, &nbands)) return;

  for(int k = 0; k < nbands; k++)
  {
    const double sigma_km1 = (k == 0) ? 0.0 : sigma_d[k - 1];
    const double lambda_peak = _band_peak_lambda(sigma_km1, sigma_d[k]);
    // implementation-plan-2.md §3.4/§4.2: _ct_fit_eval now takes fit's own
    // (frame-relative, since §4.2) sigma convention, not lambda_peak
    // directly (which would silently reintroduce the old 2*pi mismatch) and
    // not the band's own boundary sigma_d[k] (a different quantity -- the
    // boundary, not the peak). sigma_d[] is physical (pixels of the same
    // roi_in the ladder was built from), same as lambda_peak, so it needs
    // the same /long_edge conversion §4.2 applies to the ladder's own sigma.
    const double sigma_peak = lambda_peak / CT_SIGMA_TO_LAMBDA / (double)long_edge;

    double S, N;
    _ct_fit_eval(fit, sigma_peak, &S, &N);
    const double e_predicted = fmax(S + N, CT_CALIBRATION_FLOOR);

    calibration[k] = (float)CLAMP(e_module[k] / e_predicted, CT_CALIBRATION_MIN, CT_CALIBRATION_MAX);
  }
}

// §2.2: query the ladder's published SAT tables for the box the picker
// selected, feed the resulting per-rung (wavelength, energy, weight) triples
// to §2.3's `_fit_spectrum` -- primed, per §2.4, with a frame-wide noise
// estimate rather than fitting N freely -- and (§2.4) read the box's own
// sparseness back out of the same tables. box is in the pixels of
// g->ladder_roi_in, i.e. of the preview the ladder was built from.
//
// research.md §5.2: any box query is 4 lookups per rung -- this is that
// query, one held critical section covering every rung so the buffer can't
// be resized out from under it mid-query.
//
// on success, *fit holds the model and *mode which of §2.5's target shapes
// it earns (TEXTURE normally, DETAIL when A came back negligible -- "a
// self-similar area with no size"). returns FALSE only under the same
// refusals `_fit_texture_scale` always used -- too few rungs, too narrow a
// span, or nothing above the noise floor anywhere in the box; the two
// dt_control_log calls below are advisory only and never cause a refusal.
// spectrum_lambda/spectrum_energy/spectrum_nrungs (all optional, NULL to
// skip) return this box's own raw per-rung measurement -- §3.2's graph
// overlay wants it alongside the fit itself, to plot what was actually
// measured next to what the model made of it. long_edge (§4.2) is the
// ladder roi's own long edge, in the same pixels as the ladder's sigma --
// dividing by it is what makes the returned fit->tau frame-relative.
static gboolean _fit_curve_from_box(dt_iop_module_t *self, const int *const box,
                                    const float long_edge,
                                    _ct_fit_t *const fit, _ct_target_mode_t *const mode,
                                    double *const restrict spectrum_lambda,
                                    double *const restrict spectrum_energy,
                                    int *const restrict spectrum_nrungs)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  double lambda[CT_MAX_BANDS], sigma[CT_MAX_BANDS], energies[CT_MAX_BANDS], weights[CT_MAX_BANDS];
  double s1_energy[CT_MAX_BANDS];    // §2.4: Sum(|b|)/n_eff over the box, for its own sparseness
  double noise_floor[CT_MAX_BANDS];  // §2.4: frame-wide, not the box's own
  int nrungs = 0;
  double ladder_lambda0 = 0.0;       // §6.1: finest rung's own wavelength, set below

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

    // §1.1: a rung whose wavelength does not fit inside the box at least once
    // is not measuring the box's own texture -- past that size the block sum
    // is dominated by the box's offset from its surroundings (a bias, not
    // noise) and the fit will lock onto that instead. One full period is the
    // loosest defensible cut; see implementation-plan-2.md §1.1.
    const double box_w = (double)(bx1 - bx0) * CT_BLOCK;
    const double box_h = (double)(by1 - by0) * CT_BLOCK;
    const double lambda_max = fmin(box_w, box_h);
    // §6.1: the ladder's own finest rung, independent of the window above --
    // needed even when the box is too small to keep a single rung, to quote
    // the smallest box that would have worked.
    ladder_lambda0 = g->ladder_lambda[0];

    const float *const restrict buf = g->pd.buf;
    nrungs = 0;
    for(int r = 0; r < g->ladder_nrungs; r++)
    {
      if(g->ladder_lambda[r] > lambda_max) break;  // rungs run fine -> coarse

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
      const double lam = g->ladder_lambda[r];

      // §1.2: n_eff is a pixel count and is the wrong denominator for the
      // sampling term in the weight -- research.md §5.1's own error table is
      // written in terms of area/sigma^2, i.e. independent samples of the
      // rung's own period, not level pixels. Keep n_eff for the energy mean
      // (s2/n_eff is correct there) and use n_indep only for the weight.
      const double n_indep = fmax(box_w * box_h / (lam * lam), 0.25);

      lambda[nrungs] = lam;
      // §4.2: frame-relative, so it lines up with §4.1's band sigma
      sigma[nrungs] = g->ladder_sigma[r] / (double)long_edge;
      energies[nrungs] = s2 / n_eff;
      s1_energy[nrungs] = s1 / n_eff;
      weights[nrungs] = 1.0 / (CT_MODEL_ERROR * CT_MODEL_ERROR + 2.0 / n_indep);
      noise_floor[nrungs] = g->ladder_noise_floor[r];
      nrungs++;
    }
  }

  dt_iop_gui_leave_critical_section(self);

  if(!have_data)
  {
    dt_control_log(_("the preview isn't ready to measure yet -- try again in a moment"));
    return FALSE;
  }

  double peak_e = 0.0;
  for(int r = 0; r < nrungs; r++) peak_e = fmax(peak_e, energies[r]);

  // §2.4: fix N from the frame-wide block-minimum estimate rather than
  // fitting it freely -- "stabilises everything else" (research.md §5.5) and
  // stops a genuinely fine texture from getting explained away as noise.
  const double noise_prior = _ladder_estimate_noise(sigma, noise_floor, nrungs);

  // §6.1: say *which* refusal this is instead of one message covering both
  // -- "too small" and "flat" want different reactions from the user.
  _ct_fit_refusal_t refusal = CT_FIT_REFUSED_FLAT;
  if(!_fit_spectrum(sigma, energies, weights, nrungs, noise_prior, fit, &refusal))
  {
    if(refusal == CT_FIT_REFUSED_SPAN)
    {
      // the smallest box that would work at the current preview scale is
      // computable: CT_MIN_SPAN octaves' worth of the finest rung's own
      // wavelength -- the loosest lower bound §1.1's window allows.
      const double min_side = CT_MIN_SPAN * ladder_lambda0;
      dt_control_log(_("the picked area is too small to measure a detail size from -- "
                        "try at least %.0f x %.0f px"), min_side, min_side);
    }
    else
    {
      dt_control_log(_("the picked area has nothing to measure a detail size from -- "
                        "flat sky, a blown highlight and a black frame all look like this"));
    }
    return FALSE;
  }

  // §2.5/research.md §5.6: what the texture term actually delivers over the
  // measured rungs (fit->texture_peak) is negligible next to the box's own
  // peak energy: there is no sized texture here, only sensor noise and/or
  // ordinary (beta != 2) scene content -- fall back to the broad DETAIL
  // shape rather than inventing a bump. (fit->texture itself is not
  // comparable to peak_e -- see §2.3's own comment on why texture_peak
  // exists.)
  *mode = (fit->texture_peak <= peak_e * 1e-2) ? CT_TARGET_DETAIL : CT_TARGET_TEXTURE;

  // §2.4/research.md §5.9: advisory only, neither warning below refuses the
  // pick -- both just explain a result that might otherwise look like
  // nothing happened, or like an untrustworthy size.
  {
    int peak_idx = 0;
    for(int r = 1; r < nrungs; r++) if(energies[r] > energies[peak_idx]) peak_idx = r;
    double S, N;
    _ct_fit_eval(fit, sigma[peak_idx], &S, &N);
    if(S <= (S + N) * CT_NOISE_DOMINATED_FRAC)
      dt_control_log(_("the picked area looks like noise -- try raising the noise bias"));
  }

  if(*mode == CT_TARGET_TEXTURE)
  {
    const double target_sigma = sqrt(fit->tau);

    // §6.2: the fitted size sits within half an octave of the window's own
    // coarse edge (§1.1's lambda_max) -- there is no peak inside what the
    // box could see, only a rising flank, so the reported size is read off
    // the edge of the window rather than measured. Warn, don't refuse: this
    // is the honest answer, not a bad one.
    if(target_sigma >= sigma[nrungs - 1] / M_SQRT2)
      dt_control_log(_("the measured size sits at the edge of what this box can see -- "
                        "it may be larger than reported"));

    int nearest = 0;
    double best_d = DBL_MAX;
    for(int r = 0; r < nrungs; r++)
    {
      const double dist = fabs(log(sigma[r] / target_sigma));
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

  if(spectrum_lambda && spectrum_energy && spectrum_nrungs)
  {
    memcpy(spectrum_lambda, lambda, sizeof(double) * nrungs);
    memcpy(spectrum_energy, energies, sizeof(double) * nrungs);
    *spectrum_nrungs = nrungs;
  }

  return TRUE;
}

// ---------------------------------------------------------------------------
// §3.4: presets, built the same way the picker projects a target curve onto
// the nine bands (_project_to_bands) -- research.md §5.7's own "the same
// solve also gives you preset-building for free" -- but from an analytic
// shape instead of a picked area's fit. Shapes below are hand-picked
// starting points, the same way atrous.c's own shipped presets are plain
// numbers with no cited derivation; nothing here claims the empirical
// grounding phase 0's decomposition choices had.
// ---------------------------------------------------------------------------

// implementation-plan-2.md §5.2: §4.1's frame-relative formula -- H_k
// depends only on the *ratio* between sigma and lambda, so the arbitrary
// large "frame size" this used to need (to keep modify_roi_in's own "-1"
// pixel-discretisation term negligible) was never necessary, and neither
// was the term itself. scale_shift = 0: presets sit on the standard ladder.
static void _preset_nominal_sigma(float *const restrict sigma)  // CT_BANDS, finest-first
{
  int idx = 0;
  for(int k = CT_BANDS - 1; k >= 0; k--)
  {
    const double D = CT_BAND_D0 + k + 0.5;
    sigma[idx++] = (float)exp2(-(D + 1.0));
  }
}

// project target[] (evaluated on lambda_grid[]) onto the nine bands and
// write the result into p->band[], coarsest-first -- the same reversal
// color_picker_apply does at the end of its own projection. gain_lo/gain_hi
// (§4.4) is this preset's own envelope: most presets built target[] as
// 1 + strength*shape(), so their envelope is exactly [1, 1+strength] (or
// the reverse for a sub-1 preset like "soften"); "flatten spectrum" already
// clamps target[] itself, so its own bounds are what it passes here.
static gboolean _preset_apply_target(const double *const restrict lambda_grid,
                                     const double *const restrict target, const int m,
                                     const float *const restrict sigma,
                                     const float gain_lo, const float gain_hi,
                                     dt_iop_contrast_params_t *const p)
{
  float gains[CT_BANDS];
  if(!_project_to_bands(lambda_grid, target, m, sigma, CT_BANDS, NULL, gain_lo, gain_hi, gains))
    return FALSE;
  for(int k = 0; k < CT_BANDS; k++) p->band[k] = gains[CT_BANDS - 1 - k];
  return TRUE;
}

// raised-cosine bump on x in [0,1] (the graph's own coarse=0/fine=1 axis,
// via _spectrum_lambda_to_x), centred at x0 with half-width w -- zero
// outside [x0-w, x0+w], peak 1 at x0.
static double _preset_bump(const double x, const double x0, const double w)
{
  const double t = CLAMP((x - x0) / w, -1.0, 1.0);
  return 0.5 * (1.0 + cos(M_PI * t));
}

void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_contrast_params_t p;
  memset(&p, 0, sizeof(p));
  p.gain_local_contrast = 1.0f;
  p.scale_shift = 0.0f;
  p.edge_protection = 0.0f;
  p.filter_iterations = 1;
  p.noise_bias = 0.001f;
  p.decomposition = CT_DECOMPOSITION_ACCURATE;
  for(int k = 0; k < CT_BANDS; k++) p.band[k] = 1.0f;

  // implementation-plan-2.md §5.2/§4.3: sigma-native grid, frame-relative
  // (long edge = 1.0 throughout -- _spectrum_lambda_to_x's roi_long_edge
  // argument, and _preset_nominal_sigma above), matching the picker's own
  // §4.3 grid exactly.
  float sigma[CT_BANDS];
  _preset_nominal_sigma(sigma);
  double lambda_grid[CT_PROJECT_GRID], sigma_grid[CT_PROJECT_GRID], target[CT_PROJECT_GRID];
  const double lo = fmax((double)sigma[0], 1e-6) * 0.25;
  const double hi = (double)sigma[CT_BANDS - 1] * 4.0;
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    sigma_grid[j] = lo * exp2(log2(hi / lo) * (double)j / (double)(CT_PROJECT_GRID - 1));
    lambda_grid[j] = sigma_grid[j] * CT_SIGMA_TO_LAMBDA;
  }

  // "clarity": a broad boost centred mid-ladder, slightly toward the coarse
  // side -- traditional medium/large-scale local contrast.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    target[j] = 1.0 + 0.6 * _preset_bump(x, 0.35, 0.5);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.6f, &p))
    dt_gui_presets_add_generic(_("clarity"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "texture": a broader boost biased toward the fine end -- less peaked
  // than micro-contrast below, meant as a general "add texture" default.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    target[j] = 1.0 + 0.5 * _preset_bump(x, 0.7, 0.6);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.5f, &p))
    dt_gui_presets_add_generic(_("texture"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "micro-contrast": only the finest couple of bands, ramped in rather
  // than a bump, for a tighter, more surgical fine-detail boost.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    const double ramp = pow(CLAMP((x - 0.55) / 0.45, 0.0, 1.0), 1.5);
    target[j] = 1.0 + 0.8 * ramp;
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.8f, &p))
    dt_gui_presets_add_generic(_("micro-contrast"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "soften": the inverse of micro-contrast -- an edge-aware fine-detail
  // smoother, g_k < 1 toward the finest bands, untouched at the coarse end.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    const double ramp = CLAMP((x - 0.5) / 0.5, 0.0, 1.0);
    target[j] = 1.0 - 0.7 * ramp;
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 0.3f, 1.0f, &p))
    dt_gui_presets_add_generic(_("soften"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "flatten spectrum": research.md §5.6's "equalize" mode, now also the
  // live picker's CT_TARGET_EQUALIZE (§8.1) -- same _target_curve, evaluated
  // against a synthetic self-similar spectrum (beta = 2.4, research.md
  // §5.3's typical measured slope; no sized texture) rather than any
  // particular picked area's own fit, since a preset has no box to measure.
  {
    const _ct_fit_t synthetic = { .self_similar = 1.0, .beta = 2.4, .noise = 0.02,
                                  .texture = 0.0, .tau = 0.0 };
    _target_curve(&synthetic, CT_TARGET_EQUALIZE, sigma_grid, CT_PROJECT_GRID, target);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, CT_EQUALIZE_GAIN_LO, CT_EQUALIZE_GAIN_HI, &p))
    dt_gui_presets_add_generic(_("flatten spectrum"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);
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

  // implementation-plan-2.md §4.2: the ladder's sigma is in the pixels of
  // this same roi_in -- dividing by the roi's own long edge (the same
  // quantity modify_roi_in computes as S * roi_in->scale) is what makes the
  // fit's tau frame-relative, and so comparable to §4.1's frame-relative
  // band sigma below.
  const float long_edge = (float)(MAX(pipe->iwidth, pipe->iheight)) * (float)roi_in.scale;

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

  _ct_fit_t fit;
  _ct_target_mode_t mode;
  double spectrum_lambda[CT_MAX_BANDS], spectrum_energy[CT_MAX_BANDS];
  int spectrum_nrungs = 0;
  // §6.1: _fit_curve_from_box already logs a specific reason on every
  // refusal path -- nothing generic left to say here.
  if(!_fit_curve_from_box(self, box, long_edge, &fit, &mode,
                          spectrum_lambda, spectrum_energy, &spectrum_nrungs))
    return;

  // §3.2: publish this pick's own spectrum + fit for the graph's live
  // overlay -- a record of the last measurement, independent of whether the
  // picker itself is still "fresh" by the time it gets drawn.
  dt_iop_gui_enter_critical_section(self);
  memcpy(g->spectrum_lambda, spectrum_lambda, sizeof(double) * spectrum_nrungs);
  memcpy(g->spectrum_energy, spectrum_energy, sizeof(double) * spectrum_nrungs);
  g->spectrum_nrungs = spectrum_nrungs;
  g->spectrum_noise = fit.noise;
  g->spectrum_self_similar = fit.self_similar;
  g->spectrum_texture = fit.texture;
  g->spectrum_tau = fit.tau;
  g->spectrum_beta = fit.beta;
  g->spectrum_valid = TRUE;
  dt_iop_gui_leave_critical_section(self);

  // write into params and commit *before* refreshing the widget: the refresh
  // below runs under the gui-update guard and so deliberately writes nothing
  // back, which is the whole point of that guard -- it normally runs the other
  // way around, syncing widgets to params that have already changed.
  dt_iop_contrast_params_t *p = self->params;

  // §2.5/research.md §5.6: the picker sets shape, never strength -- with one
  // exception, the same one blackwhite's picker uses to turn its filter on:
  // a shape with no strength behind it (gain still at its neutral default)
  // would be invisible, so raise it first.
  if(p->gain_local_contrast == 1.0f) p->gain_local_contrast = 1.5f;

  // implementation-plan-2.md §4.1: the nominal per-band boundary sigma,
  // frame-relative (sigma / long edge) and finest-first (idx 0) to match
  // `_project_to_bands`'s H_k derivation. The band ladder is frame-relative
  // by construction (node k's nominal wavelength is S * 2^-(D0+k+shift)) --
  // the "-1" pixel-discretisation term and the roi_in.scale factor belong to
  // modify_roi_in, where a sigma has to come out in the pixels of an actual
  // buffer; the projection has no such need, and at preview scale the old
  // formula put sigma[0] at exactly 0 (a zero H_0 column, the finest band
  // permanently unreachable from a pick) and spanned 21 octaves, most of it
  // sub-pixel garbage.
  float sigma[CT_BANDS];
  {
    int idx = 0;
    for(int k = CT_BANDS - 1; k >= 0; k--)
    {
      const double D = CT_BAND_D0 + k + 0.5 + p->scale_shift;
      sigma[idx++] = (float)exp2(-(D + 1.0));  // sigma / long edge; diameter/2, no pixel term
    }
  }

  // implementation-plan-2.md §4.3: dense log-sigma grid spanning the node
  // ladder itself, padded two octaves either side (sigma[0]*0.25 ..
  // sigma[CT_BANDS-1]*4, 13 octaves total, none of it below the finest
  // band) so the projection sees each end band's full response rather than
  // a truncated one. _target_curve/_ct_fit_eval are evaluated directly on
  // sigma_grid; _project_to_bands' H_k needs a real wavelength, so
  // lambda_grid is sigma_grid scaled by CT_SIGMA_TO_LAMBDA.
  double lambda_grid[CT_PROJECT_GRID], sigma_grid[CT_PROJECT_GRID];
  double shape[CT_PROJECT_GRID], target[CT_PROJECT_GRID];
  const double lo = fmax((double)sigma[0], 1e-6) * 0.25;
  const double hi = (double)sigma[CT_BANDS - 1] * 4.0;
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    sigma_grid[j] = lo * exp2(log2(hi / lo) * (double)j / (double)(CT_PROJECT_GRID - 1));
    lambda_grid[j] = sigma_grid[j] * CT_SIGMA_TO_LAMBDA;
  }

  _target_curve(&fit, mode, sigma_grid, CT_PROJECT_GRID, shape);
  // §8.1: CT_TARGET_EQUALIZE's shape[] is already the bounded absolute
  // target curve (research.md §5.6/§3.4's "flatten spectrum") -- use it as
  // target[] as-is, not lerped between 1 and master the way TEXTURE/DETAIL's
  // [0,1] shape is.
  const gboolean is_equalize = (mode == CT_TARGET_EQUALIZE);
  for(int j = 0; j < CT_PROJECT_GRID; j++)
    target[j] = is_equalize ? shape[j] : 1.0 + ((double)p->gain_local_contrast - 1.0) * shape[j];

  // §3.1: how much of its own linear H_k each band actually delivered over
  // this same box, last time the module's own bands were measured there --
  // uncalibrated (all 1s) if that measurement isn't available yet.
  float calibration[CT_BANDS];
  _compute_band_calibration(self, box, long_edge, &fit, calibration);

  // §4.4: the envelope the target curve itself was built to -- TEXTURE/
  // DETAIL's is 1 + (master-1)*shape, shape in [0,1]; EQUALIZE's is its own
  // fixed clamp (§8.1, same as the preset). No band should leave it.
  const float gain_lo = is_equalize ? CT_EQUALIZE_GAIN_LO : fminf(1.0f, p->gain_local_contrast);
  const float gain_hi = is_equalize ? CT_EQUALIZE_GAIN_HI : fmaxf(1.0f, p->gain_local_contrast);

  float gains[CT_BANDS];  // finest-first, matching sigma[] above
  if(!_project_to_bands(lambda_grid, target, CT_PROJECT_GRID, sigma, CT_BANDS, calibration,
                        gain_lo, gain_hi, gains))
  {
    dt_control_log(_("could not fit a curve to the picked area"));
    return;
  }

  for(int k = 0; k < CT_BANDS; k++)
    p->band[k] = gains[CT_BANDS - 1 - k];  // back to coarsest-first param order

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

// ---------------------------------------------------------------------------
// §3.2: the live spectrum overlay -- research.md's own suggestion (§5.6,
// implementation-plan.md §1.4 step 4/toneequal's inset histogram) to draw
// the frame-wide ladder as the graph's background at all times, and the
// last picked box's own spectrum plus its fitted model on top of it once a
// pick has landed. Makes the picker legible instead of magic: the data was
// already being measured (§2.1/§3.1), this just puts it on screen.
// ---------------------------------------------------------------------------

// energies span orders of magnitude across ordinary images (phase0-band-
// energy.md's own tables), so the y axis here is log, normalised to
// whichever curve on screen has the higher peak -- purely relative shape,
// no fixed absolute meaning.
#define CT_SPECTRUM_LOG_RANGE 10.0  // stops of dynamic range shown below the on-screen peak

static float _spectrum_energy_to_y(const double energy, const double peak)
{
  if(energy <= 0.0 || peak <= 0.0) return 0.0f;
  return CLAMP((float)(1.0 + log2(energy / peak) / CT_SPECTRUM_LOG_RANGE), 0.0f, 1.0f);
}

// the frame-wide ladder's own spectrum: one (lambda, energy) point per rung,
// energy = S2/n over the *whole* ladder grid. Since a summed-area table is
// zero-padded on its low side (§2.1's _ladder_build_sat), the frame total is
// simply the table's own opposite corner -- no subtraction needed, unlike a
// box query.
static gboolean _spectrum_frame_wide(dt_iop_module_t *self,
                                     double *const restrict lambda,
                                     double *const restrict energy,
                                     int *const restrict nrungs)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;
  gboolean ok = FALSE;

  dt_iop_gui_enter_critical_section(self);
  const size_t sat_w = g->pd.width, sat_h = g->pd.height;
  const size_t comps = g->pd.components;
  if(g->pd.buf && sat_w > 1 && sat_h > 1 && g->ladder_nrungs > 0
     && comps == (size_t)(2 * g->ladder_nrungs))
  {
    const double nblocks = (double)(sat_w - 1) * (double)(sat_h - 1);
    const float *const restrict buf = g->pd.buf;
    const size_t corner = (sat_h - 1) * sat_w + (sat_w - 1);
    *nrungs = g->ladder_nrungs;
    for(int r = 0; r < g->ladder_nrungs; r++)
    {
      const double step = g->ladder_step[r];
      const double n_per_block = fmax(1.0, (CT_BLOCK / step) * (CT_BLOCK / step));
      const double n_eff = fmax(nblocks * n_per_block, 1.0);
      lambda[r] = g->ladder_lambda[r];
      energy[r] = (double)buf[corner * comps + 2 * r] / n_eff;
    }
    ok = TRUE;
  }
  dt_iop_gui_leave_critical_section(self);
  return ok;
}

// draw one (lambda[], energy[]) polyline, in the current cairo source, over
// the graph's plotting area.
static void _draw_spectrum_curve(cairo_t *cr, const int width, const int height,
                                 const double *const restrict lambda,
                                 const double *const restrict energy,
                                 const int n, const double roi_long_edge, const double peak)
{
  if(n < 1) return;
  gboolean started = FALSE;
  for(int r = 0; r < n; r++)
  {
    const float x = _spectrum_lambda_to_x(lambda[r], roi_long_edge) * width;
    const float y = height * (1.0f - _spectrum_energy_to_y(energy[r], peak));
    if(!started) { cairo_move_to(cr, x, y); started = TRUE; }
    else cairo_line_to(cr, x, y);
  }
  cairo_stroke(cr);
}

static void _draw_spectrum_overlay(cairo_t *cr, dt_iop_module_t *self,
                                   const int width, const int height)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  double frame_lambda[CT_MAX_BANDS], frame_energy[CT_MAX_BANDS];
  int frame_nrungs = 0;
  const gboolean have_frame = _spectrum_frame_wide(self, frame_lambda, frame_energy, &frame_nrungs);

  dt_iop_gui_enter_critical_section(self);
  const gboolean have_pick = g->spectrum_valid;
  const int pick_nrungs = g->spectrum_nrungs;
  double pick_lambda[CT_MAX_BANDS], pick_energy[CT_MAX_BANDS];
  double fit_noise = 0.0, fit_self_similar = 0.0, fit_texture = 0.0, fit_tau = 0.0, fit_beta = 2.0;
  if(have_pick)
  {
    memcpy(pick_lambda, g->spectrum_lambda, sizeof(double) * pick_nrungs);
    memcpy(pick_energy, g->spectrum_energy, sizeof(double) * pick_nrungs);
    fit_noise = g->spectrum_noise;
    fit_self_similar = g->spectrum_self_similar;
    fit_texture = g->spectrum_texture;
    fit_tau = g->spectrum_tau;
    fit_beta = g->spectrum_beta;
  }
  const double roi_long_edge = MAX(g->ladder_roi_in.width, g->ladder_roi_in.height);
  dt_iop_gui_leave_critical_section(self);

  if(!have_frame && !have_pick) return;

  double peak = 0.0;
  for(int r = 0; r < frame_nrungs; r++) peak = fmax(peak, frame_energy[r]);
  for(int r = 0; r < pick_nrungs; r++) peak = fmax(peak, pick_energy[r]);
  if(peak <= 0.0) return;

  cairo_save(cr);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));

  if(have_frame)
  {
    cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                             darktable.bauhaus->graph_border.green,
                             darktable.bauhaus->graph_border.blue, 0.8);
    _draw_spectrum_curve(cr, width, height, frame_lambda, frame_energy, frame_nrungs,
                         roi_long_edge, peak);
  }

  if(have_pick)
  {
    cairo_set_source_rgba(cr, darktable.bauhaus->color_fill.red,
                             darktable.bauhaus->color_fill.green,
                             darktable.bauhaus->color_fill.blue, 0.9);
    _draw_spectrum_curve(cr, width, height, pick_lambda, pick_energy, pick_nrungs,
                         roi_long_edge, peak);

    // the fitted S(lambda) + N(lambda) model, sampled densely across the
    // picked box's own measured range, dashed to read as "model" rather
    // than "measurement" next to the polyline above.
    const _ct_fit_t fit = { .noise = fit_noise, .self_similar = fit_self_similar,
                            .texture = fit_texture, .tau = fit_tau, .beta = fit_beta };
    const double dashes[2] = { DT_PIXEL_APPLY_DPI(4.0), DT_PIXEL_APPLY_DPI(3.0) };
    cairo_set_dash(cr, dashes, 2, 0.0);
    gboolean started = FALSE;
    const double lo = pick_lambda[0], hi = pick_lambda[pick_nrungs - 1];
    for(int j = 0; j <= CT_GRAPH_RES; j++)
    {
      const double lambda = lo * exp2(log2(hi / fmax(lo, 1e-6)) * (double)j / (double)CT_GRAPH_RES);
      double S, N;
      // implementation-plan-2.md §3.2: _ct_fit_eval takes sigma, via
      // CT_SIGMA_TO_LAMBDA -- §4.2 adds a further frame-relative conversion
      // once fit->tau itself becomes frame-relative.
      _ct_fit_eval(&fit, lambda / CT_SIGMA_TO_LAMBDA, &S, &N);
      const float x = _spectrum_lambda_to_x(lambda, roi_long_edge) * width;
      const float y = height * (1.0f - _spectrum_energy_to_y(S + N, peak));
      if(!started) { cairo_move_to(cr, x, y); started = TRUE; }
      else cairo_line_to(cr, x, y);
    }
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
  }

  cairo_restore(cr);
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

  // 2b. §6.3: shade the coarse end of the axis the last pick's own window
  // (§1.1) could not see -- coarse sits at x=0, same axis the unresolvable
  // shading above uses. The honest version of 6.1/6.2's refusal/warning:
  // this is *why* the pick answered what it did. The coarsest rung that
  // survived the window (published as the last entry of g->spectrum_lambda,
  // already the windowed set per §1.3) stands in for the window's own
  // cutoff -- close enough, since nothing coarser than it was ever queried.
  {
    dt_iop_gui_enter_critical_section(self);
    const gboolean have_pick_window = g->spectrum_valid && g->spectrum_nrungs > 0;
    const double pick_coarsest_lambda =
      have_pick_window ? g->spectrum_lambda[g->spectrum_nrungs - 1] : 0.0;
    const double window_roi_long_edge = MAX(g->ladder_roi_in.width, g->ladder_roi_in.height);
    dt_iop_gui_leave_critical_section(self);

    if(have_pick_window)
    {
      const float x1 = _spectrum_lambda_to_x(pick_coarsest_lambda, window_roi_long_edge) * width;
      if(x1 > 0.0f)
      {
        cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                                 darktable.bauhaus->graph_border.green,
                                 darktable.bauhaus->graph_border.blue, 0.4);
        cairo_rectangle(cr, 0, 0, x1, height);
        cairo_fill(cr);
      }
    }
  }

  // 3. baseline at gain 1.0
  const float baseline_y = height * (1.0f - 1.0f / CT_GRAPH_Y_MAX);
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
  dt_draw_line(cr, 0, baseline_y, width, baseline_y);
  cairo_stroke(cr);

  // 4. the measured spectrum (§3.2): frame-wide ladder always in the
  // background, the last pick's own spectrum + fitted model on top of it.
  _draw_spectrum_overlay(cr, self, width, height);

  // 5. the curve: monotone cubic through the nine nodes
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
  dt_preview_data_alloc(&g->band_pd, self);  // §3.1: the module's own per-band tables
  g->band_pd.components = 2 * CT_BANDS;  // fixed forever, unlike the ladder's nrungs -- no resize dance

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
  dt_preview_data_free(&g->band_pd);  // §3.1
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
