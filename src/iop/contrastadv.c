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
#include "common/luminance_mask.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "dtgtk/paint.h"
#include "dtgtk/togglebutton.h"
#include "dtgtk/expander.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "libs/lib.h"
#include "common/iop_group.h"

#ifdef _OPENMP
#include <omp.h>
#endif

DT_MODULE_INTROSPECTION(2, dt_iop_contrast_params_t)

#define CT_BANDS 9          // detail levels 2..10, one node per octave
#define CT_BAND_D0 2.0f     // detail level of the coarsest node

// the graph: nodes run coarse (left) to fine (right), one per octave, so the
// x axis is simply k/CT_BANDS; the y axis is gain, soft-ranged to
// CT_GRAPH_Y_MAX to match the sliders' own soft range (gui_init) even though
// the hard range (band[]'s $MAX) reaches higher -- a node dragged past the
// top just rides the edge, exactly like the slider it drives.
#define CT_GRAPH_Y_MAX 2.0f
#define CT_GRAPH_RES 64      // curve points sampled between nodes, per implementation-plan.md §1.4

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

// the journey of one pick between two threads: color_picker_apply() arms it on
// the GUI thread, process() claims and measures it on the pipe thread, and the
// preview-pipe-finished handler commits the result back on the GUI thread. the
// claim is what keeps a second drag from being swallowed: the measurement
// takes tens of milliseconds, and without it an in-flight result would land on
// top of a newer request and strand it.
typedef enum _ct_auto_state_t
{
  CT_AUTO_IDLE = 0,   // nothing pending
  CT_AUTO_REQUESTED,  // a pick is waiting for a preview pass to measure it
  CT_AUTO_MEASURING,  // process() has claimed that pick and is working on it
  CT_AUTO_READY       // a result is waiting for the GUI thread to commit
} _ct_auto_state_t;

// what the measurement had to say for itself
typedef enum _ct_auto_result_t
{
  CT_AUTO_MEASURED = 0, // auto_detail_level holds a size
  CT_AUTO_NOTHING,      // too small an area, or nothing in it to size
  CT_AUTO_NO_MEMORY
} _ct_auto_result_t;

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

  // cross-thread hand-off for the detail level area picker. every read and
  // write of these three goes through dt_iop_gui_enter/leave_critical_section.
  _ct_auto_state_t auto_state;
  _ct_auto_result_t auto_result;
  float auto_detail_level;
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
// measuring the detail level off a picked area
// ---------------------------------------------------------------------------
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
// what a picked area is short of is ladder. the hump is located by its flanks,
// so the ladder has to reach past the texture on both sides, and every octave
// of it costs the mirrored border twice over -- while a pyramid that decimates
// costs, on top of that, the strip it crops and half of what is left. on a
// full frame none of that is felt; on a box drawn over one sleeve it is the
// whole budget, and the ladder used to run out while the texture was still
// ahead of it, which reads back as a texture that is smaller than it is. so
// the two economies are separated: decimate while decimating is free, and
// after that keep the pixels and climb by doubling the ladder itself. that is
// what lets a 24-pixel box be measured at all, and what keeps a 120-pixel one
// honest about texture four times coarser than it could previously report.
//
// the whole thing runs on log2 luminance, which is the space the module's own
// detail extraction works in, so the measurement sees exactly the signal that
// will be boosted -- including the shadow-noise suppression from the noise
// bias slider, since it is measured on the same biased luminance buffer.

#define CT_SCALES_PER_OCTAVE 3    // DoG bands per octave of the ladder
#define CT_SIGMA_BASE 1.2f        // finest sigma of the ladder, in analysis pixels
#define CT_MAX_OCTAVES 12
#define CT_MAX_BANDS (CT_MAX_OCTAVES * CT_SCALES_PER_OCTAVE)
#define CT_KERNEL_RADIUS_MAX 32
// smallest interior a level may still be measured on, per axis
#define CT_MIN_INTERIOR ((size_t)8)
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
// the fit's other grid: the self-similar floor as a fraction of the hump's
// height, scanned over 2^-CT_FIT_RATIO_FLOOR .. 2^(STEPS - FLOOR)
#define CT_FIT_RATIO_STEPS 16
#define CT_FIT_RATIO_FLOOR 12.0
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

// the rectangle an area picker asked us to measure, inside the single-channel
// luminance buffer process() has already built for its own use.
typedef struct _ct_region_t
{
  const float *lum;      // start of the buffer, one float per pixel
  size_t stride;         // buffer width, in pixels
  size_t x0, y0;         // region origin within the buffer
  size_t width, height;  // region size
} _ct_region_t;

static int _gauss_kernel(const float sigma, float *const restrict kern)
{
  const int r = MIN((int)ceilf(3.0f * sigma), CT_KERNEL_RADIUS_MAX);
  float sum = 0.0f;
  for(int i = -r; i <= r; i++)
  {
    const float w = expf(-0.5f * (float)(i * i) / (sigma * sigma));
    kern[i + r] = w;
    sum += w;
  }
  for(int i = 0; i < 2 * r + 1; i++) kern[i] /= sum;
  return r;
}

// whole-sample symmetric reflection: ... 2 1 0 1 2 ... rather than the usual
// edge replication. this matters more than it looks. replication continues a
// texture with a flat plateau, which is a structure the picture does not
// contain: the two rungs of a DoG resolve that plateau differently and the
// band picks up energy that isn't there, biasing every level's border. a
// mirrored texture is still a texture with the same statistics, so the bias
// is second order and a modest margin is enough to absorb what remains.
static inline int _mirror(int x, const int n)
{
  if(n < 2) return 0;
  const int period = 2 * n - 2;
  x = x % period;
  if(x < 0) x += period;
  return (x < n) ? x : period - x;
}

// the sigma of ladder rung s, in the pixels of the level it is measured on
static inline double _rung_sigma(const double unit, const int s)
{
  return unit * exp2((double)s / CT_SCALES_PER_OCTAVE);
}

// how far the mirrored border reaches into each rung, in that level's own
// pixels: the strip that has to be dropped from the energy sums, and from the
// top rung before it is handed down to the next octave.
//
// this is the kernel radius of the single blur that produces the rung from its
// octave's base image, which is the whole reason each rung is blurred straight
// from that base rather than from the rung below it. a chain of blurs reaches
// as far as the *sum* of its radii -- radii add where sigmas combine in
// quadrature -- and on a small picked area that sum is most of what there is
// to measure. going back to the base each time costs a few more kernel taps
// and buys back roughly a third of the usable width.
//
// so this is the reach from the level's *base*, and it is the reach from the
// picked area itself only where the two coincide: on the first octave, and on
// every octave after a decimation, which crops the reached strip away. a
// ladder climbing in place (see _measure_detail_scale) inherits its base's
// reach uncounted, and a couple of pixels per side of already-mirrored texture
// leak back into its sums. that is deliberate. mirroring is exact under
// composition -- symmetric reflection commutes with symmetric convolution, so
// the level is the honest blur of the mirror-extended area rather than
// anything corrupted -- and what remains is the second-order bias of measuring
// a locally even field, worth two or three hundredths of an octave where
// counting the inherited reach costs three or four tenths in lost ladder.
//
// radii are computed from the same _gauss_kernel() the blurs themselves use,
// so they cannot drift away from them.
static void _ladder_radii(const double unit,
                          const double base_sigma,
                          int *const restrict radii)
{
  float kern[2 * CT_KERNEL_RADIUS_MAX + 1];

  for(int s = 0; s <= CT_SCALES_PER_OCTAVE; s++)
  {
    const double sigma = _rung_sigma(unit, s);
    const double var = sigma * sigma - base_sigma * base_sigma;
    radii[s] = var > 0.0 ? _gauss_kernel((float)sqrt(var), kern) : 0;
  }
}

// separable Gaussian with mirrored borders. sigma is bounded here -- a level
// climbs at most one octave above its own base, and a ladder that has to climb
// in place instead of decimating (see _measure_detail_scale) is stopped once
// its kernel reaches CT_KERNEL_RADIUS_MAX -- so a short explicit kernel is
// both exact and cheap.
static void _gauss_blur(const float *const restrict src,
                        float *const restrict dst,
                        float *const restrict tmp,
                        const size_t width,
                        const size_t height,
                        const float sigma)
{
  float kern[2 * CT_KERNEL_RADIUS_MAX + 1];
  const int r = _gauss_kernel(sigma, kern);

  // the horizontal pass needs a wrap decision per tap, so it is split into
  // the two borders, where _mirror() earns its integer division, and the
  // interior, where by construction no tap can leave the row. the vertical
  // pass below needs no such treatment: it hoists the row pointer out of the
  // inner loop already, so it mirrors once per row and tap rather than once
  // per pixel and tap.
  const size_t edge = MIN((size_t)r, width);
  const size_t inner_end = (width > (size_t)r) ? width - (size_t)r : edge;

  DT_OMP_FOR()
  for(size_t j = 0; j < height; j++)
  {
    const float *const row = src + j * width;
    float *const out = tmp + j * width;

    for(size_t i = 0; i < edge; i++)
    {
      float acc = 0.0f;
      for(int t = -r; t <= r; t++)
        acc += kern[t + r] * row[_mirror((int)i + t, (int)width)];
      out[i] = acc;
    }
    for(size_t i = edge; i < inner_end; i++)
    {
      float acc = 0.0f;
      for(int t = -r; t <= r; t++)
        acc += kern[t + r] * row[i + t];
      out[i] = acc;
    }
    for(size_t i = inner_end; i < width; i++)
    {
      float acc = 0.0f;
      for(int t = -r; t <= r; t++)
        acc += kern[t + r] * row[_mirror((int)i + t, (int)width)];
      out[i] = acc;
    }
  }

  DT_OMP_FOR()
  for(size_t j = 0; j < height; j++)
  {
    float *const out = dst + j * width;
    memset(out, 0, width * sizeof(float));
    for(int t = -r; t <= r; t++)
    {
      const float *const row = tmp + (size_t)_mirror((int)j + t, (int)height) * width;
      const float weight = kern[t + r];
      for(size_t i = 0; i < width; i++) out[i] += weight * row[i];
    }
  }
}

// produce ladder rung s from the octave's base image
static void _ladder_rung(const float *const restrict base,
                         float *const restrict dst,
                         float *const restrict scratch,
                         const size_t width,
                         const size_t height,
                         const double unit,
                         const double base_sigma,
                         const int s)
{
  const double sigma = _rung_sigma(unit, s);
  const double var = sigma * sigma - base_sigma * base_sigma;

  if(var > 0.0)
    _gauss_blur(base, dst, scratch, width, height, (float)sqrt(var));
  else
    memcpy(dst, base, width * height * sizeof(float));  // already there
}

// mean squared difference of two neighbouring rungs, over the interior only
static double _band_energy(const float *const restrict a,
                           const float *const restrict b,
                           const size_t width,
                           const size_t height,
                           const size_t margin)
{
  const size_t iw = width - 2 * margin;
  const size_t ih = height - 2 * margin;
  double sum = 0.0;

  DT_OMP_FOR(reduction(+ : sum))
  for(size_t j = margin; j < height - margin; j++)
  {
    for(size_t i = margin; i < width - margin; i++)
    {
      const double d = (double)a[j * width + i] - (double)b[j * width + i];
      sum += d * d;
    }
  }

  return sum / (double)(iw * ih);
}

// plain subsampling by two, of the interior only. the source has just been
// blurred to twice the ladder's base sigma, which puts the new grid's Nyquist
// frequency about four sigma out, so there is nothing left up there to alias.
//
// dropping the reached border here rather than carrying it along is what keeps
// the pyramid honest. whatever border bias a level has left, passing it down would
// halve it and then have the next octave's own blurs widen it again -- a
// recurrence that settles well inside the interior the energies are summed
// over, so the first band of each octave comes out inflated and the ladder
// develops a sawtooth that swamps the peak it is supposed to reveal. cropping
// makes every octave start clean instead, at the cost of that reach in pixels
// per side per octave.
static void _decimate2_interior(const float *const restrict src,
                                const size_t width,
                                const size_t margin,
                                const size_t new_width,
                                const size_t new_height,
                                float *const restrict dst)
{
  DT_OMP_FOR()
  for(size_t j = 0; j < new_height; j++)
    for(size_t i = 0; i < new_width; i++)
      dst[j * new_width + i] = src[(margin + 2 * j) * width + margin + 2 * i];
}

// the size the ladder is best explained by, as a wavelength in the pixels the
// ladder was measured in, or 0 if it does not describe a texture at all.
//
// the argmax is a hopeless estimator here and it is worth being explicit about
// why: the normalized-Laplacian curve of a texture with a characteristic size
// is extremely broad -- an octave either side of its peak it has only fallen
// by about a sixth -- so a few percent of sampling noise on one rung walks the
// argmax a whole octave, and in practice it does, constantly. what actually
// locates the size is not the flat top but the two flanks, which are steep
// (the energy climbs as t^2 below the peak and falls as 1/t above it) and span
// decades. so fit the whole curve's shape, and fit it in log energy, where
// those decades carry their proper weight and each rung's roughly constant
// *relative* error becomes a constant absolute one.
//
// the shape is available in closed form for the simplest thing that counts as
// "texture of a size": white noise smoothed to sigma_t, whose spectrum is
// exp(-4 pi^2 f^2 sigma_t^2). integrating the normalized-Laplacian response
// against it gives A * t^2 / (t + tau)^3 with tau = sigma_t^2, a single hump
// peaking at t = 2 tau, whose height turns out to depend only on the texture's
// contrast and not on its size -- so two textures of equal contrast and
// different size compete on equal terms. anything self-similar underneath --
// the 1/f^2 part of the picture, which is most of it -- contributes a constant
// instead, which is precisely what the gamma = 1 normalization was chosen to
// arrange.
//
// that leaves three parameters: the hump's size tau, its height, and the floor
// under it. only the first two of the three matter to the answer, and in log
// energy the height is a pure vertical offset, solved in closed form by a mean.
// so scan tau and the floor-to-height ratio over a grid and take the pair with
// the smallest residual.
//
// the rungs are not equally trustworthy -- one measured on the last, nearly
// exhausted level of a small area rests on a handful of independent samples
// where one from the full width rests on thousands -- so every sum below is
// weighted by how many samples the rung actually had.
static double _fit_texture_scale(const double *const restrict sigmas,
                                 const double *const restrict energies,
                                 const double *const restrict weights,
                                 const int nbands)
{
  // the smallest area that can be measured at all covers exactly one octave,
  // so this comparison is met on the nose there and is given a rounding's
  // worth of slack rather than being left to turn on an ulp.
  if(nbands < CT_MIN_BANDS
     || sigmas[nbands - 1] < CT_MIN_SPAN * sigmas[0] * (1.0 - 1e-9))
    return 0.0;

  double n = 0.0, peak_e = 0.0;
  for(int i = 0; i < nbands; i++)
  {
    n += weights[i];
    peak_e = fmax(peak_e, energies[i]);
  }

  // nothing there at any scale: a blank sky, a blown highlight, a black frame
  if(peak_e <= CT_FLAT_ENERGY) return 0.0;

  // a rung that has fallen this far below the peak carries no information
  // about anything except its own rounding error, and in log space it would
  // otherwise dominate the residual outright.
  double y[CT_MAX_BANDS];
  for(int i = 0; i < nbands; i++)
    y[i] = log(fmax(energies[i], peak_e * CT_ENERGY_FLOOR));

  // candidate texture sizes: from half the finest rung to the coarsest one.
  // the top of that range puts the hump's peak just past the end of the
  // ladder, which is as far as the rising flank alone can honestly be pushed.
  const double lo = sigmas[0] * 0.5;
  const double hi = sigmas[nbands - 1];
  const int steps = MAX((int)(CT_FIT_STEPS_PER_OCTAVE * log2(hi / lo)), 1);

  double best_residual = DBL_MAX;
  double best_sigma_t = 0.0;

  for(int q = 0; q <= steps; q++)
  {
    const double sigma_t = lo * exp2((double)q / CT_FIT_STEPS_PER_OCTAVE);
    const double tau = sigma_t * sigma_t;

    double shape[CT_MAX_BANDS];
    for(int i = 0; i < nbands; i++)
    {
      const double t = sigmas[i] * sigmas[i];
      const double d = t + tau;
      // the hump, scaled to unit height so that the ratio scanned below is
      // directly the floor's height as a fraction of the hump's
      shape[i] = 6.75 * tau * t * t / (d * d * d);
    }

    for(int m = 0; m <= CT_FIT_RATIO_STEPS; m++)
    {
      const double ratio = exp2((double)m - CT_FIT_RATIO_FLOOR);

      double model[CT_MAX_BANDS];
      double offset = 0.0;
      for(int i = 0; i < nbands; i++)
      {
        model[i] = log(shape[i] + ratio);
        offset += weights[i] * (y[i] - model[i]);
      }
      offset /= n;  // the hump's height, in closed form

      double residual = 0.0;
      for(int i = 0; i < nbands; i++)
      {
        const double d = y[i] - model[i] - offset;
        residual += weights[i] * d * d;
      }

      if(residual < best_residual)
      {
        best_residual = residual;
        best_sigma_t = sigma_t;
      }
    }
  }

  if(best_sigma_t <= 0.0) return 0.0;

  // note what happens when the area is self-similar -- dry grass, sand, plain
  // sensor noise -- and has no one size at all: the ladder comes out flat, no
  // position of the hump explains it better than any other, and the fit
  // settles at the fine end of the grid because that is where the hump's own
  // rising flank is shortest and disturbs the flat fit least. reporting the
  // finest measurable size for such an area is the right answer anyway: it is
  // the setting that boosts the texture without reaching up into structure
  // that isn't there, and it is the one that cannot produce halos.

  // the hump peaks at t = 2 tau, i.e. at sigma = sqrt(2) * sigma_t, and the
  // normalized Laplacian of a sinusoid of wavelength L peaks at
  // sigma = L / (pi * sqrt(2)) -- so the wavelength this stands for is simply
  // 2 * pi * sigma_t.
  return 2.0 * M_PI * best_sigma_t;
}

// returns the dominant texture wavelength over the region, in pixels of the
// buffer it was measured on; 0 if there is nothing measurable there, and a
// negative value if it could not get the memory to look.
static double _measure_detail_scale(const _ct_region_t *const region)
{
  const size_t w0 = region->width, h0 = region->height;
  const size_t npixels = w0 * h0;

  {
    // the finest band of the first octave is the least this can possibly do
    int radii[CT_SCALES_PER_OCTAVE + 1];
    _ladder_radii(CT_SIGMA_BASE, 0.0, radii);
    const size_t smallest = 2 * (size_t)radii[1] + CT_MIN_INTERIOR;
    if(w0 < smallest || h0 < smallest) return 0.0;
  }

  // the first three are handed around between the roles below, so they are
  // left unqualified: at any moment they still name three distinct buffers.
  float *base = dt_alloc_align_float(npixels);
  float *buf_lo = dt_alloc_align_float(npixels);
  float *buf_hi = dt_alloc_align_float(npixels);
  float *const restrict scratch = dt_alloc_align_float(npixels);
  if(!base || !buf_lo || !buf_hi || !scratch)
  {
    dt_free_align(base);
    dt_free_align(buf_lo);
    dt_free_align(buf_hi);
    dt_free_align(scratch);
    return -1.0;
  }

  // the module boosts differences of log2 luminance, so that is what has to be
  // measured. the buffer handed in already carries the noise bias, exactly as
  // extract_details() sees it.
  const float *const restrict lum = region->lum;
  const size_t stride = region->stride;
  const size_t x0 = region->x0, y0 = region->y0;

  double mean_ev = 0.0;

  DT_OMP_FOR(reduction(+ : mean_ev))
  for(size_t j = 0; j < h0; j++)
  {
    float *const out = base + j * w0;
    const float *const row = lum + (y0 + j) * stride + x0;
    double rowsum = 0.0;
    for(size_t i = 0; i < w0; i++)
    {
      out[i] = log2f(fmaxf(row[i], NORM_MIN));
      rowsum += out[i];
    }
    mean_ev += rowsum;
  }
  mean_ev /= (double)npixels;

  // subtract the area's own level. everything downstream is a difference of
  // blurs and so ignores this offset mathematically -- but not numerically:
  // the bands being measured are a few thousandths of an EV riding on an
  // absolute level that is wherever the exposure happens to put it, and
  // leaving it in makes the answer depend on that exposure through float
  // rounding alone. centering pins the working range around zero and makes
  // the measurement exactly invariant to how the image is exposed.
  const float offset = (float)mean_ev;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
    base[k] -= offset;

  double sigmas[CT_MAX_BANDS], energies[CT_MAX_BANDS], weights[CT_MAX_BANDS];
  int nbands = 0;
  size_t cw = w0, ch = h0;

  // the ladder's base sigma in the *current buffer's* pixels, and how many
  // level-0 pixels one of those covers. decimating leaves the first unchanged
  // and doubles the second; climbing without decimating does the opposite.
  double unit = CT_SIGMA_BASE;
  double step = 1.0;
  gboolean base_is_raw = TRUE;

  for(int octave = 0; octave < CT_MAX_OCTAVES; octave++)
  {
    // octave zero's base is the raw region, so its rungs carry the whole blur
    // up from nothing; afterwards the base always arrives at exactly `unit`.
    const double base_sigma = base_is_raw ? 0.0 : unit;

    int radii[CT_SCALES_PER_OCTAVE + 1];
    _ladder_radii(unit, base_sigma, radii);

    float *lo = buf_lo, *hi = buf_hi;
    _ladder_rung(base, lo, scratch, cw, ch, unit, base_sigma, 0);

    int produced = 0;
    for(int s = 0; s < CT_SCALES_PER_OCTAVE; s++)
    {
      // a band is bounded by the coarser of its two rungs. the radii grow
      // with s, so once one band does not fit, neither does any above it.
      const size_t margin = (size_t)radii[s + 1];
      if(radii[s + 1] >= CT_KERNEL_RADIUS_MAX
         || cw < 2 * margin + CT_MIN_INTERIOR
         || ch < 2 * margin + CT_MIN_INTERIOR)
        break;

      _ladder_rung(base, hi, scratch, cw, ch, unit, base_sigma, s + 1);

      energies[nbands] = _band_energy(lo, hi, cw, ch, margin);
      // the DoG's own peak frequency sits within a percent of the one the
      // geometric mean of its two sigmas predicts, so label the band with that
      // and carry it back to level-0 pixels.
      const double sigma_here = sqrt(_rung_sigma(unit, s) * _rung_sigma(unit, s + 1));
      sigmas[nbands] = sigma_here * step;

      // and how much this rung is worth to the fit, as one over the variance
      // its log carries.
      //
      // one part of that is sampling: a mean of squares over n independent
      // samples has a relative variance of 2/n, and the samples of a band stop
      // being independent about sigma pixels apart, so the interior it was
      // summed over is worth roughly its area divided by sigma squared.
      //
      // the other part does not shrink with n at all. the hump being fitted is
      // an idealization, and a real texture departs from it by some percent no
      // matter how many pixels are averaged. without that floor the weights
      // span two orders of magnitude across a deep pyramid -- each octave has
      // four times fewer samples than the one below -- and the fine rungs
      // drown out the coarse flank entirely, which is precisely the half of
      // the curve that says a texture is large.
      const double interior = (double)(cw - 2 * margin) * (double)(ch - 2 * margin);
      const double samples = fmax(interior / (sigma_here * sigma_here), 1.0);
      weights[nbands] =
        1.0 / (CT_MODEL_ERROR * CT_MODEL_ERROR + 2.0 / samples);
      nbands++;
      produced++;

      float *const swap = lo; lo = hi; hi = swap;
    }

    // the ladder is bounded by the *region*: a small picked area cannot show
    // that the texture keeps growing past its own edge, so it saturates at a
    // finer answer. picking large is what makes a coarse answer available.
    // stop as soon as an octave cannot be completed -- without its top rung
    // there is nothing to carry forward.
    if(produced < CT_SCALES_PER_OCTAVE) break;

    // lo now holds the rung at twice the base sigma, which is the next
    // octave's base image either way. the only question is how to get there.
    const size_t crop = (size_t)radii[CT_SCALES_PER_OCTAVE];
    const size_t nw = cw > 2 * crop ? (cw - 2 * crop) / 2 : 0;
    const size_t nh = ch > 2 * crop ? (ch - 2 * crop) / 2 : 0;

    // what the child would need to be worth having: enough room for its
    // *coarsest* band, not just its finest. handing down a level that dies
    // half way through its octave is worse than not decimating at all.
    int child_radii[CT_SCALES_PER_OCTAVE + 1];
    _ladder_radii(unit, unit, child_radii);
    const size_t child_min =
      2 * (size_t)child_radii[CT_SCALES_PER_OCTAVE] + CT_MIN_INTERIOR;

    if(nw >= child_min && nh >= child_min)
    {
      // decimating is what keeps this linear in the area, and on anything of
      // a decent size it is free: the image is already blurred to twice the
      // base sigma, so its Nyquist frequency is four sigma out and there is
      // nothing left up there to alias.
      _decimate2_interior(lo, cw, crop, nw, nh, base);
      cw = nw;
      ch = nh;
      step *= 2.0;
    }
    else
    {
      // but on a small area it is ruinous -- the crop and the halving together
      // are the difference between another octave of ladder and none -- and
      // the speed it buys is worth nothing there anyway. so keep the pixels
      // and climb by doubling the ladder itself instead. the rungs get wider
      // kernels and the borders reach further, which is what eventually stops
      // this, but it stops one or two octaves later than decimating would.
      float *const previous = base;
      if(lo == buf_lo) { base = buf_lo; buf_lo = previous; }
      else             { base = buf_hi; buf_hi = previous; }
      unit *= 2.0;
    }

    base_is_raw = FALSE;
  }

  dt_free_align(base);
  dt_free_align(buf_lo);
  dt_free_align(buf_hi);
  dt_free_align(scratch);

  return _fit_texture_scale(sigmas, energies, weights, nbands);
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

  for(int k = 0; k < d->nbands; k++)
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

// measure the area the picker was dragged over and hand the resulting detail
// level back to the GUI thread.
static void _auto_detail_level(dt_iop_module_t *self,
                               dt_iop_contrast_gui_data_t *g,
                               const float *const restrict luminance,
                               const dt_iop_roi_t *const roi_in,
                               const dt_dev_pixelpipe_iop_t *const piece)
{
  // the picker hands back a mean over the area, which is of no use here -- we
  // need a scale-space ladder over the actual pixels -- so take the box it drew
  // and do the measuring ourselves. the region defaults to the whole frame,
  // which is also the fallback if the box cannot be mapped into this module's
  // coordinates.
  _ct_region_t region = { .lum = luminance,
                          .stride = roi_in->width,
                          .x0 = 0, .y0 = 0,
                          .width = roi_in->width,
                          .height = roi_in->height };

  const dt_colorpicker_sample_t *const sample =
    darktable.lib->proxy.colorpicker.primary_sample;
  int box[4];
  if(sample
     && sample->size == DT_LIB_COLORPICKER_SIZE_BOX
     && !dt_color_picker_box(self, roi_in, sample, PIXELPIPE_PICKER_INPUT, box))
  {
    region.x0 = box[0];
    region.y0 = box[1];
    region.width = box[2] - box[0];
    region.height = box[3] - box[1];
  }

  const double wavelength = _measure_detail_scale(&region);

  _ct_auto_result_t result = wavelength < 0.0 ? CT_AUTO_NO_MEMORY : CT_AUTO_NOTHING;
  float level = 0.0f;
  if(wavelength > 0.0)
  {
    // invert what modify_roi_in() does with a node's placement: node D's
    // window is 2^-D * max_size * roi->scale pixels wide, so asking for a
    // window exactly one texture wavelength wide -- the shortest box average
    // that removes that texture from the base layer completely, and so hands
    // all of it to the high pass without also dragging in anything coarser --
    // gives the D below, which the caller folds into scale_shift.
    //
    // the scale factor cancels the fact that this was measured on the preview:
    // what comes out is a fraction of the frame and is carried unchanged to
    // full resolution. what does not cancel is that the preview cannot resolve
    // texture finer than a few of its own pixels, which is what bounds the
    // fine end of the answer.
    const double max_size = MAX(piece->iwidth, piece->iheight);
    const double scale = fmax((double)roi_in->scale, 1e-6);
    level = (float)CLAMP(log2(max_size * scale / wavelength), 0.0, 15.0);
    result = CT_AUTO_MEASURED;
  }

  dt_iop_gui_enter_critical_section(self);
  // only publish if this is still the pick we claimed. if a newer one arrived
  // while we were measuring it has already asked for its own preview pass, so
  // leave it armed and drop what we just computed rather than committing a
  // size for an area the user has moved on from; and if focus left the module
  // in the meantime, drop it as well.
  if(g->auto_state == CT_AUTO_MEASURING)
  {
    g->auto_detail_level = level;
    g->auto_result = result;
    g->auto_state = CT_AUTO_READY;
  }
  dt_iop_gui_leave_critical_section(self);
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

  // An area measurement needs to see a whole, contiguous frame, so skip it
  // while the pipe hands us one tile at a time -- the request simply stays
  // pending and is served by the next untiled preview run.
  if(g && self->dev->gui_attached
     && dt_pipe_is_preview(piece->pipe)
     && !piece->pipe->tiling)
  {
    // claim the pending pick before the slow part, so that a second drag
    // arriving mid-measurement stays armed instead of being overwritten
    dt_iop_gui_enter_critical_section(self);
    const gboolean claimed = g->auto_state == CT_AUTO_REQUESTED;
    if(claimed) g->auto_state = CT_AUTO_MEASURING;
    dt_iop_gui_leave_critical_section(self);

    if(claimed)
      _auto_detail_level(self, g, luminance, roi_in, piece);
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

static void _preview_pipe_finished_callback(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(!g) return;

  dt_iop_gui_enter_critical_section(self);
  const gboolean ready = g->auto_state == CT_AUTO_READY;
  const _ct_auto_result_t result = g->auto_result;
  const float level = g->auto_detail_level;
  if(ready) g->auto_state = CT_AUTO_IDLE;
  dt_iop_gui_leave_critical_section(self);

  if(!ready) return;

  if(result == CT_AUTO_NO_MEMORY)
  {
    dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
    return;
  }
  if(result != CT_AUTO_MEASURED)
  {
    dt_control_log(_("the picked area is too small, or has nothing in it to measure a detail size from"));
    return;
  }

  // write into params and commit *before* refreshing the widget: the refresh
  // below runs under the gui-update guard and so deliberately writes nothing
  // back, which is the whole point of that guard -- it normally runs the other
  // way around, syncing widgets to params that have already changed.
  //
  // this only positions the ladder (scale_shift); it does not yet reshape the
  // bands around the measured size -- that reshaping is Phase 1.7.
  dt_iop_contrast_params_t *p = self->params;
  const float d = CLAMP(level, CT_BAND_D0, CT_BAND_D0 + CT_BANDS - 1);
  p->scale_shift = CLAMP(d - roundf(d), -0.5f, 0.5f);

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

// the area picker lands here once the pipe has sampled its box. we ignore the
// sampled color itself -- what we want is the pixels underneath it, which only
// process() can see -- so this just asks for one more preview pass to measure
// on.
//
// the picker machinery only calls us when the box has actually moved, so
// applying the result (which commits history and re-runs the pipe) cannot
// bounce straight back in here: the measurement runs once per drag.
void color_picker_apply(dt_iop_module_t *self,
                        GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(!g || picker != g->scale_shift) return;

  dt_iop_gui_enter_critical_section(self);
  g->auto_state = CT_AUTO_REQUESTED;
  dt_iop_gui_leave_critical_section(self);

  dt_dev_reprocess_preview(self->dev, self->iop_order);
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(in) return;

  dt_iop_color_picker_reset(self, TRUE);

  // and drop any pick that has not been served yet. a request only gets
  // measured on a preview pass that reaches this module untiled, which may
  // never come -- the pipe may tile, or the module may be switched off first.
  // left armed, it would be picked up by whatever preview pass happens next,
  // reading whatever box the color picker holds by then, and would rewrite the
  // detail level and push a history item with no user action behind it.
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(!g) return;

  dt_iop_gui_enter_critical_section(self);
  g->auto_state = CT_AUTO_IDLE;
  dt_iop_gui_leave_critical_section(self);
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
  g->auto_state = CT_AUTO_IDLE;
  g->auto_result = CT_AUTO_NOTHING;

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished_callback);
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
       "fast: not implemented yet -- currently behaves identically to accurate."));

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

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_draw_curve_destroy(g->curve);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
