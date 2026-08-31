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

typedef enum dt_iop_details_display_t
{
  DT_LC_MASK_OFF = -1,
  DT_LC_MASK_LOCAL = 0,
  DT_LC_MASK_LAST = 1
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

// Compute smoothed luminance mask using edge-aware filters
__DT_CLONE_TARGETS__
static inline void compute_luminance_and_mask(const float *const restrict in,
                                              float *const restrict luminance,
                                              float *const restrict smoothed_luminance,
                                              const dt_iop_roi_t *const roi_in,
                                              const dt_iop_contrast_data_t *const d)
{
  size_t width = (size_t)roi_in->width;
  size_t height = (size_t)roi_in->height;
  const size_t npixels = width * height;

  // First compute pixel-wise luminance (no boost) and add noise bias
  luminance_mask(in, luminance, width, height, DT_TONEEQ_NORM_2, 1.0f, 0.0f, 1.0f);
  const float noise_bias = d->noise_bias;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    luminance[k] += noise_bias;
  }

  // Then apply the smoothing filter on a copy
  memcpy(smoothed_luminance, luminance, npixels * sizeof(float));

  fast_eigf_surface_blur(smoothed_luminance, width, height,
                         d->radius_local, d->feathering, d->iterations,
                         DT_GF_BLENDING_LINEAR, 1.0f,
                         0.0f, NORM_MIN, 4.0f);
}

// Extract logarithmic high pass detail in log space (EV):
// How much brighter/darker is this pixel compared to the smooth version
__DT_CLONE_TARGETS__
static inline float extract_details(const float luminance_pixel,
                                   const float luminance_smoothed,
                                   const float noise_bias)
{
  const float log_pixel = log2f(fmaxf(luminance_pixel, NORM_MIN));
  const float log_smoothed = log2f(fmaxf(luminance_smoothed, NORM_MIN));

  const float noise_power = noise_bias * noise_bias;
  const float combined_power = luminance_smoothed * luminance_smoothed;
  const float weiner_gain = fmaxf(combined_power - noise_power, 0.0f) / fmaxf(combined_power, NORM_MIN);
  return weiner_gain * fmaxf(fminf(log_pixel - log_smoothed, 5.0f), -5.0f);
}

// Apply local contrast enhancement
// The detail (local contrast) is the log-space difference between pixel luminance
// and smoothed luminance. Boosting this difference amplifies local details.
__DT_CLONE_TARGETS__
static inline void apply_local_contrast(const float *const restrict in,
                                        const float *const restrict luminance_pixel,
                                        const float *const restrict luminance_smoothed,
                                        float *const restrict out,
                                        const dt_iop_roi_t *const roi_in,
                                        const dt_iop_contrast_data_t *const d)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;
  const float gain_local = (d->gain_local_contrast - 1.0f);
  const float noise_bias = d->noise_bias;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    // High pass detail in log space (EV):
    // How much brighter/darker is this pixel compared to the smooth version
    const float local_ev = extract_details(luminance_pixel[k], luminance_smoothed[k], noise_bias);

    // Correction as the scaled ev difference
    const float correction_ev = gain_local * local_ev;

    // Apply correction in linear space
    const float multiplier = exp2f(correction_ev);;
    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
    out[4 * k + 3] = in[4 * k + 3];
  }
}

/*
 Display the detail mask (difference between pixel and smoothed luminance)
 Output is a grayscale image normalized to [0, 1] where:
 - 0.5 = no local detail (pixel matches neighborhood)
 - < 0.5 = pixel darker than neighborhood
 - > 0.5 = pixel brighter than neighborhood
 */
__DT_CLONE_TARGETS__
static inline void display_local_mask(const float *const restrict luminance_pixel,
                                      const float *const restrict luminance_smoothed,
                                      float *const restrict out,
                                      const dt_iop_roi_t *const roi_in,
                                      const dt_iop_contrast_data_t *const d)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;
  const float noise_bias = d->noise_bias;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float local_ev = extract_details(luminance_pixel[k], luminance_smoothed[k], noise_bias);

    // Detail in log space, mapped to [0, 1] for display
    // Detail range roughly [-2, +2] EV mapped to [0, 1]
    const float intensity = local_ev / sqrtf(local_ev * local_ev + 1.0f) * 0.5f + 0.5f; // Smooth mapping to [0, 1]

    // Set all RGB channels to the same intensity (grayscale)
    for_each_channel(c)
    {
      out[4 * k + c] = intensity;
    }
    // Full opacity
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
    // invert what modify_roi_in() does with the parameter: it builds a box of
    // contrast_scale * max_size * roi->scale pixels, and detail_level is minus
    // the log2 of contrast_scale. so asking for a window exactly one texture
    // wavelength wide -- the shortest box average that removes that texture
    // from the base layer completely, and so hands all of it to the high pass
    // without also dragging in anything coarser -- gives the level below.
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

  const float *const restrict in = (float *const)ivoid;
  float *const restrict out = (float *const)ovoid;

  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const size_t npixels = width * height;

  float *restrict luminance_pixel = dt_alloc_align_float(npixels);
  float *restrict luminance_smoothed_local = dt_alloc_align_float(npixels);

  if(!luminance_pixel || !luminance_smoothed_local)
  {
    dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
    dt_free_align(luminance_pixel);
    dt_free_align(luminance_smoothed_local);
    return;
  }

  compute_luminance_and_mask(in, luminance_pixel, luminance_smoothed_local, roi_in, d);

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
      _auto_detail_level(self, g, luminance_pixel, roi_in, piece);
  }

  // Display output
  if(g && g->details_display != DT_LC_MASK_OFF && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL))
  {
    display_local_mask(luminance_pixel, luminance_smoothed_local, out, roi_in, d);
    piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
  }
  else
  {
    apply_local_contrast(in, luminance_pixel, luminance_smoothed_local, out, roi_in, d);
  }

  dt_free_align(luminance_pixel);
  dt_free_align(luminance_smoothed_local);
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
  dt_iop_contrast_params_t *p = self->params;
  p->detail_level = level;

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
  if(!g || picker != g->detail_level) return;

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

static void show_details_callback(GtkWidget *togglebutton, dt_iop_module_t *self)
{
  // early return if blend module is already displaying a mask
  if(self->request_mask_display)
  {
    dt_control_log(_("cannot display masks when the blending mask is displayed"));
    dt_bauhaus_widget_set_quad_active(GTK_WIDGET(togglebutton), FALSE);
    return;
  }

  DT_GUARD_GUI_UPDATE();
  dt_iop_request_focus(self);
  // Activate the module if it wasn't
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  g->details_display = DT_LC_MASK_OFF;

  const gboolean toggle_is_active = dt_bauhaus_widget_get_quad_active(GTK_WIDGET(togglebutton));
  if(toggle_is_active)
  {
    if(togglebutton == g->gain_local_contrast) g->details_display = DT_LC_MASK_LOCAL;
  }

  dt_bauhaus_widget_set_quad_active(GTK_WIDGET(g->gain_local_contrast), g->details_display == DT_LC_MASK_LOCAL);
  dt_iop_refresh_center(self);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = IOP_GUI_ALLOC(contrast);
  g->details_display = DT_LC_MASK_OFF;
  g->auto_state = CT_AUTO_IDLE;
  g->auto_result = CT_AUTO_NOTHING;

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished_callback);

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
                             _("visualize details adjusted by the local constrast"));

  // Filter settings section
  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "filter settings")));

  g->detail_level = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                        dt_bauhaus_slider_from_params(self, "detail_level"));
  dt_bauhaus_slider_set_soft_range(g->detail_level, 2.0, 10.0);
  gtk_widget_set_tooltip_text(g->detail_level,
     _("detail level adjusted by the local contrast.\n"
       "higher = more contrast boost in finer details\n"
       "lower = more contrast boost in coarser details"));
  dt_bauhaus_widget_set_quad_tooltip
    (g->detail_level,
     _("pick an area: measure how large the texture in it actually is, and set\n"
       "the detail level so that texture -- and nothing coarser -- is what gets\n"
       "boosted.\n"
       "click to use the whole frame, then drag on the image to work from the\n"
       "subject that matters instead. a small box cannot report structure\n"
       "larger than itself, so pick over as much of the texture as you want\n"
       "counted.\n"
       "an area with nothing in it to enhance -- clear sky, an out-of-focus\n"
       "background -- is declined rather than guessed at. if a pick over deep\n"
       "shadow comes back at the finest setting it is reading sensor noise;\n"
       "raise the noise bias below and pick again."));


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

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
