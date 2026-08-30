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

#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <gtk/gtk.h>

#include "bauhaus/bauhaus.h"
#include "common/box_filters.h"
#include "common/imagebuf.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/tiling.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"

// gain applied to the local chroma detail before it is added to the grey mix.
// that detail is a projection of the very same RGB pixel the grey mix is made
// of, so it carries the same units and this is a plain dimensionless factor --
// no dependency on exposure or on the working profile's scaling.
#define BLACKWHITE_CHROMA_CONTRAST_GAIN 0.5f


DT_MODULE_INTROSPECTION(1, dt_iop_blackwhite_params_t)

typedef struct dt_iop_blackwhite_params_t
{
  gboolean filter;        // $DEFAULT: FALSE $DESCRIPTION: "color filter"
  float hue;              // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 0.0 $DESCRIPTION: "hue"
  float chroma;           // $MIN: 0.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "chroma"
  float chroma_contrast;  // $MIN: 0.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "strength"
  float eye_response;     // $MIN: 0.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "human vision"
  // as a percentage of the image diagonal, so it means the same thing whatever
  // the camera's resolution and survives being carried across in a preset
  float detail_radius;    // $MIN: 0.05 $MAX: 10.0 $DEFAULT: 1.0 $DESCRIPTION: "detail radius"
} dt_iop_blackwhite_params_t;

typedef struct dt_iop_blackwhite_gui_data_t
{
  GtkWidget *filter;
  GtkWidget *swatch;
  GtkWidget *hue;
  GtkWidget *chroma;
  GtkWidget *chroma_contrast;
  GtkWidget *detail_radius;
  GtkWidget *eye_response;
  GtkWidget *auto_button;

  // cross-thread hand-off for the "auto" button: process() fills in auto_params
  // and flips auto_state to 2 when it sees state 1 on the preview pipe; the
  // preview-pipe-finished signal handler (GUI thread) then applies auto_params
  // and resets state to 0. protected by dt_iop_gui_enter/leave_critical_section.
  int auto_state; // 0: idle, 1: computation requested, 2: result ready to apply
  dt_iop_blackwhite_params_t auto_params;
} dt_iop_blackwhite_gui_data_t;

typedef struct dt_iop_blackwhite_data_t
{
  gboolean filter;
  float hue;
  float chroma;
  float chroma_contrast;
  float eye_response;
  float detail_radius;
} dt_iop_blackwhite_data_t;

typedef struct dt_iop_blackwhite_global_data_t
{
  int kernel_blackwhite;
} dt_iop_blackwhite_global_data_t;


const char *name()
{
  return _("black and white");
}

int default_group()
{
  return IOP_GROUP_COLOR | IOP_GROUP_EFFECTS;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self,
     _("convert the image to black & white,\n"
       "optionally through a virtual colored lens filter"),
     _("creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

// classic Wratten-style B&W lens filters, approximated as (hue, chroma) pairs in
// our R=0°/G=120°/B=240° mixing model. not extracted from any LUT/spectral data --
// just the well-known real-world color/strength of each filter.
typedef struct _bw_filter_preset_t
{
  const char *name;
  float hue;
  float chroma;
} _bw_filter_preset_t;

// hue/chroma fitted (least-squares harmonic projection of the real Kodak B-3
// spectral transmission data through CIE1931/D65/sRGB, exactly as computed by
// compute_channel_multipliers() in the reference generator at
// ~/Pictures/LUTs/wratten filters/filters.py) onto our grey[c] = 1/3 +
// chroma*cos(hue - hue_c) model -- not hand-guessed.
//
// the fit was made in sRGB primaries, which is precisely what this module's
// reference space is (see _work_to_reference()), so these numbers are always
// read in the space they were made in, whatever working profile the pipe runs.
// a wider working space needs no refit either: weight triples summing to 1 are
// a two-parameter family and (hue, chroma) covers all of them, so the fitted
// mix is carried over exactly rather than approximated.
//
// the D65 here is the illuminant the transmission was integrated against -- a
// property of the modeled filter ("what a Wratten 25 does to a daylight-lit
// scene"), not of the pipe, and unrelated to the D50 connection space the
// weights are transported through below. the weights are normalized to sum 1
// in any case, so the illuminant only tilts their relative distribution.
static const _bw_filter_preset_t _bw_filter_presets[] = {
  { N_("Wratten 3 light yellow"),        62.8f, 0.183f },
  { N_("Wratten 8 yellow"),              57.7f, 0.300f },
  { N_("Wratten 9 deep yellow"),         52.2f, 0.319f },
  { N_("Wratten 11 yellow-green"),      103.5f, 0.315f },
  { N_("Wratten 12 deep yellow"),        48.0f, 0.340f },
  { N_("Wratten 15 deep yellow"),        39.4f, 0.356f },
  { N_("Wratten 16 orange"),             30.8f, 0.382f },
  { N_("Wratten 21 orange"),             18.4f, 0.445f },
  { N_("Wratten 22 deep orange"),        10.7f, 0.511f },
  { N_("Wratten 23A light red"),          5.2f, 0.578f },
  { N_("Wratten 24 red-orange"),          2.0f, 0.629f },
  { N_("Wratten 25 red"),                 1.0f, 0.647f },
  { N_("Wratten 26 red"),                 0.5f, 0.657f },
  { N_("Wratten 29 deep red"),            0.0f, 0.667f },
  { N_("Wratten 32 magenta"),           284.7f, 0.318f },
  { N_("Wratten 33 strong magenta"),    341.8f, 0.446f },
  { N_("Wratten 34A violet"),           252.3f, 0.485f },
  { N_("Wratten 38A blue"),             219.0f, 0.349f },
  { N_("Wratten 39 blue glass"),        243.1f, 0.588f },
  { N_("Wratten 44 light cyan"),        190.4f, 0.326f },
  { N_("Wratten 44A minus red"),        208.4f, 0.353f },
  { N_("Wratten 47 blue"),              241.0f, 0.586f },
  { N_("Wratten 47A light blue"),       234.8f, 0.504f },
  { N_("Wratten 47B deep blue"),        243.3f, 0.597f },
  { N_("Wratten 58 green"),             116.2f, 0.554f },
  { N_("Wratten 61 deep green"),        117.9f, 0.576f },
  { N_("Wratten 74 dark green"),        118.9f, 0.643f },
  { N_("Wratten 80A blue"),             232.2f, 0.230f },
  { N_("Wratten 80B blue"),             229.5f, 0.185f },
  { N_("Wratten 80C blue"),             227.0f, 0.125f },
  { N_("Wratten 80D blue"),             225.9f, 0.065f },
  { N_("Wratten 81 warming"),            42.6f, 0.012f },
  { N_("Wratten 81A warming"),           41.3f, 0.024f },
  { N_("Wratten 81B warming"),           39.8f, 0.035f },
  { N_("Wratten 81C warming"),           36.7f, 0.048f },
  { N_("Wratten 81D warming"),           36.2f, 0.061f },
  { N_("Wratten 81EF warming"),          38.4f, 0.076f },
  { N_("Wratten 82 cooling"),           221.8f, 0.013f },
  { N_("Wratten 82A cooling"),          224.0f, 0.030f },
  { N_("Wratten 82B cooling"),          225.1f, 0.048f },
  { N_("Wratten 82C cooling"),          222.5f, 0.068f },
  { N_("Wratten 90 monochrome viewing"), 28.2f, 0.392f },
  { N_("Wratten 98 deep blue"),         243.2f, 0.599f },
  { N_("Wratten 99 deep green"),        111.1f, 0.530f },
};

void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_blackwhite_params_t p = { .filter = FALSE, .hue = 0.f, .chroma = 0.f,
                                   .chroma_contrast = 0.f, .eye_response = 0.f,
                                   .detail_radius = 1.f };
  dt_gui_presets_add_generic(_("panchromatic (no filter)"), self->op,
                             self->version(), &p, sizeof(p), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);

  for(size_t i = 0; i < G_N_ELEMENTS(_bw_filter_presets); i++)
  {
    const _bw_filter_preset_t *f = &_bw_filter_presets[i];
    p.filter = TRUE;
    p.hue = f->hue;
    p.chroma = f->chroma;
    dt_gui_presets_add_generic(_(f->name), self->op,
                               self->version(), &p, sizeof(p), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);
  }
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_blackwhite_params_t *p = (dt_iop_blackwhite_params_t *)p1;
  dt_iop_blackwhite_data_t *d = piece->data;

  d->filter = p->filter;
  d->hue = p->hue;
  d->chroma = p->chroma;
  d->chroma_contrast = p->chroma_contrast;
  d->eye_response = p->eye_response;
  d->detail_radius = p->detail_radius;

  // the chroma contrast pass needs a neighborhood that only the CPU path
  // implements, so don't let the pipe try OpenCL for it in the first place
  // (returning an error from process_cl would work too, but only after
  // needlessly shuffling the buffers to the GPU and back).
  if(d->chroma_contrast > 0.f)
    piece->process_cl_ready = FALSE;
}

// R/G/B weights of a color filter of the given hue in our R=0/G=120/B=240
// model, without the 1/3 pedestal. these sum to exactly zero, so applied on
// their own they extract a pure chrominance signal: any neutral pixel, at any
// brightness, projects to 0.
static inline void _chroma_axis_weights(const float hue_rad, dt_aligned_pixel_t axis)
{
  for(int c = 0; c < 3; c++)
    axis[c] = cosf(hue_rad - c * (2.f * M_PI_F / 3.f));
  axis[3] = 0.f;
}

// the module's reference space: linear Rec.709 (sRGB primaries).
//
// a filter written as a set of R/G/B channel weights only means something
// relative to a set of primaries -- the same numbers read against Rec.2020's,
// which sit out on the edge of the spectral locus, describe a considerably
// stronger filter than they do against Rec.709's. the presets above were
// fitted in sRGB primaries, so that is what a (hue, chroma) pair means here,
// and the working profile the user happens to have picked must not change it.
//
// no pixel is converted for this. everything this module does to a pixel is a
// linear form w.RGB, and a linear form transforms as a covector,
//
//     w_ref . RGB_ref = w_ref . (M RGB_work) = (M^T w_ref) . RGB_work
//
// so the whole correction is one 3x3 applied to the weights, once per pipe
// run: the per-pixel loops and the OpenCL kernel (which just takes the weights
// as a uniform) never learn about it. Rec.709 coordinates are never
// materialized either, so a color outside its gamut simply carries negative
// components through the same linear form instead of being clipped into it.

// M = (XYZ_D50 -> Rec.709) * (working RGB -> XYZ_D50), row major.
//
// both halves are D50-referenced -- darktable takes matrix_in from the ICC
// colorant tags, which lcms has already Bradford-adapted to the D50 connection
// space, and dt's own Rec.709 matrix is the D50-adapted one -- so M maps the
// working space's white exactly onto Rec.709's white. that is why the D65
// whitepoint of the primaries needs no adaptation step here, and why
// sum(w) == 1 survives the transform: neutral in, neutral out still holds by
// construction, whatever profile the pipe runs in.
//
// a profile we cannot read a matrix from falls back to the identity, i.e. to
// taking the pipe to already be in the reference space.
static void _work_to_reference(const dt_iop_order_iccprofile_info_t *const work_profile,
                               dt_colormatrix_t M)
{
  memset(M, 0, sizeof(dt_colormatrix_t));
  M[0][0] = M[1][1] = M[2][2] = 1.f;

  if(!work_profile
     || !dt_is_valid_colormatrix(work_profile->matrix_in[0][0]))
    return;

  dt_colormatrix_t xyz_to_ref;
  memset(xyz_to_ref, 0, sizeof(dt_colormatrix_t));
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      xyz_to_ref[i][j] = xyz_to_srgb_transposed[j][i];

  mat3SSEmul(M, xyz_to_ref, work_profile->matrix_in);
}

// re-express channel weights given in the reference space so that they compute
// the same thing from working-space pixels: w_work = M^T w_ref.
static void _reference_to_work(const dt_colormatrix_t M,
                               const dt_aligned_pixel_t w_ref,
                               dt_aligned_pixel_t w_work)
{
  for(int j = 0; j < 3; j++)
    w_work[j] = w_ref[0] * M[0][j] + w_ref[1] * M[1][j] + w_ref[2] * M[2][j];
  w_work[3] = 0.f;
}

// the eye's photopic luminance sensitivity in the reference space: the Y row
// of its RGB->XYZ matrix, which is the same construction Rec.709 "luma" comes
// from, and sums to 1 by definition (white has Y = 1) so it stays a pure
// re-weighting of the panchromatic mix and never changes overall brightness.
//
// this is the one quantity here that needed no anchoring: Y is Y in any space,
// and carrying these weights through _reference_to_work() reproduces the
// working profile's own matrix_in[1] exactly. it is the filter axis that had
// to be pinned to a set of primaries.
static inline void _reference_luminance(dt_aligned_pixel_t lum)
{
  for(int c = 0; c < 3; c++)
    lum[c] = sRGB_to_xyz_transposed[c][1];
  lum[3] = 0.f;
}

// the base panchromatic response (flat 1/3, 1/3, 1/3) blended toward the eye's
// actual photopic luminance sensitivity, then perturbed by the color filter on
// top of that -- so filter presets stay layerable regardless of how much "human
// vision" weighting is dialed in.
//
// built entirely in the reference space, including the flat pedestal, and
// handed back in working-space coordinates: the mix a given set of sliders
// produces is then one and the same conversion whatever profile the pipe runs
// in, rather than something that quietly changes with it.
//
// both perturbations are sum-preserving by construction (the luminance weights
// sum to 1, the filter weights sum to 0), and so is the transform out of the
// reference space, so the mix always maps a neutral input to itself; the
// normalization is only a guard against a degenerate profile.
static void _compute_grey_mix(const dt_iop_blackwhite_data_t *const d,
                              const dt_iop_order_iccprofile_info_t *const work_profile,
                              dt_aligned_pixel_t grey)
{
  dt_aligned_pixel_t ref = { 1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 0.f };

  if(d->eye_response > 0.f)
  {
    dt_aligned_pixel_t lum;
    _reference_luminance(lum);
    for(int c = 0; c < 3; c++)
      ref[c] += d->eye_response * (lum[c] - ref[c]);
  }

  if(d->filter)
  {
    dt_aligned_pixel_t axis;
    _chroma_axis_weights(deg2radf(d->hue), axis);
    for(int c = 0; c < 3; c++)
      ref[c] += d->chroma * axis[c];
  }

  const float norm = ref[0] + ref[1] + ref[2];
  if(isfinite(norm) && fabsf(norm - 1.f) > 1e-6f && fabsf(norm) > 1e-6f)
  {
    for(int c = 0; c < 3; c++) ref[c] /= norm;
  }

  dt_colormatrix_t M;
  _work_to_reference(work_profile, M);
  _reference_to_work(M, ref, grey);
}

// the color direction a grey conversion is mathematically blind to.
//
// a mix is a set of weights w with sum(w) == 1. feed it a pure chrominance
// difference d -- a color difference with sum(d) == 0 -- and it returns
// sum(w*d), which in the (x, y) chroma plane of our R=0/G=120/B=240 model is
// the linear functional
//
//     sum_c w[c]*d[c] = (2/3) * (x*G_x + y*G_y),
//     G_x = (3*w[0] - 1)/2,   G_y = (sqrt(3)/2) * (w[1] - w[2])
//
// so the mix responds only to the component along G, and maps *every* color
// difference perpendicular to G to exactly zero. that perpendicular is what
// chroma contrast has to work on. it follows from the colorimetry alone: no
// user setting, no image statistic, no measurement pass.
//
// like the mix itself this is built in the reference space, and the hue it
// returns is a reference-space hue: _chroma_axis_weights() turns it into
// reference-space weights, which _reference_to_work() then carries into the
// pipe's own coordinates.
//
// G is taken from the eye's own luminance response, rotated by the color filter
// when one is active -- never from the module's actual mix. a flat 1/3 average
// is a technical default rather than a model of vision, and being blind to the
// whole chroma plane it expresses no preference at all; "which color
// differences look different but render the same" is a perceptual question, so
// photopic luminance is what answers it. that also means the axis is always
// well defined, whatever the filter and human vision sliders are set to.
//
// of the two perpendicular directions we take the warmer one, so a patch warmer
// than its surroundings renders lighter. that agrees both with the classic
// red/orange filter look and with the Helmholtz-Kohlrausch effect, which makes
// reds read lighter than their luminance while cyans stay put.
static float _blind_axis_hue(const dt_iop_blackwhite_data_t *const d)
{
  dt_aligned_pixel_t w;
  _reference_luminance(w);

  if(d->filter)
  {
    dt_aligned_pixel_t f;
    _chroma_axis_weights(deg2radf(d->hue), f);
    for(int c = 0; c < 3; c++) w[c] += d->chroma * f[c];
  }

  const float gx = 0.5f * (3.f * w[0] - 1.f);
  const float gy = 0.5f * sqrtf(3.f) * (w[1] - w[2]);

  return atan2f(gy, gx) - M_PI_F / 2.f;
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_blackwhite_data_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

// box radius of the chroma contrast low-pass: a percentage of the image
// diagonal, converted to pixels at whatever resolution this piece of the pipe
// actually runs at. same convention as soften, so a setting means the same
// thing on any camera rather than being tied to a pixel count.
static int _chroma_contrast_radius(const dt_dev_pixelpipe_iop_t *const piece,
                                   const dt_iop_roi_t *const roi)
{
  const dt_iop_blackwhite_data_t *const d = piece->data;
  const float w = piece->iwidth * piece->iscale;
  const float h = piece->iheight * piece->iscale;
  const float full = dt_fast_hypotf(w, h) * d->detail_radius / 100.f;
  const int radius =
    MAX(1, (int)ceilf(full * roi->scale / piece->iscale));
  // a box filter wider than the image itself buys nothing and only risks
  // upsetting the filter's internal scanline handling
  return MIN(radius, (int)MAX(1, MIN(roi->width, roi->height) / 2));
}

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  const dt_iop_blackwhite_data_t *const d = piece->data;

  int overlap = 0;
  if(d->chroma_contrast > 0.f)
  {
    // dt_box_mean() applies BOX_ITERATIONS box passes, so a tile has to carry
    // far more context than one radius: use the same box-to-gaussian
    // equivalence the other box_mean users (highpass, soften) rely on.
    const int radius = _chroma_contrast_radius(piece, roi_in);
    const float sigma = sqrtf((radius * (radius + 1) * BOX_ITERATIONS + 2) / 3.0f);
    overlap = (int)ceilf(3.0f * sigma);
  }

  tiling->factor = 2.0f + 0.25f + 0.05f; // in + out + detail buffer + slice for dt_box_mean
  tiling->factor_cl = 2.0f; // OpenCL only ever runs the plain mix (see commit_params)
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = overlap;
  tiling->align = 1;
}

// mean squared difference of the blind-axis projection between pairs of pixels
// a given distance apart, averaged over the horizontal and vertical directions.
// note the projection weights sum to zero, so constant offsets drop out and
// this measures purely how much the *color* changes over that distance.
static double _axis_variogram(const float *const restrict in,
                              const size_t width,
                              const size_t height,
                              const dt_aligned_pixel_t axis,
                              const size_t lag)
{
  double sum = 0.0;
  size_t count = 0;

  if(width > lag)
  {
    double hsum = 0.0;
    DT_OMP_FOR(reduction(+ : hsum))
    for(size_t j = 0; j < height; j++)
    {
      const float *const row = in + 4 * j * width;
      for(size_t i = 0; i + lag < width; i++)
      {
        const float *const a = row + 4 * i;
        const float *const b = row + 4 * (i + lag);
        const float d = (b[0] - a[0]) * axis[0]
                      + (b[1] - a[1]) * axis[1]
                      + (b[2] - a[2]) * axis[2];
        hsum += (double)d * d;
      }
    }
    sum += hsum;
    count += height * (width - lag);
  }

  if(height > lag)
  {
    double vsum = 0.0;
    const size_t rows = height - lag; // OpenMP needs a loop-invariant bound
    DT_OMP_FOR(reduction(+ : vsum))
    for(size_t j = 0; j < rows; j++)
    {
      const float *const ra = in + 4 * j * width;
      const float *const rb = in + 4 * (j + lag) * width;
      for(size_t i = 0; i < width; i++)
      {
        const float *const a = ra + 4 * i;
        const float *const b = rb + 4 * i;
        const float d = (b[0] - a[0]) * axis[0]
                      + (b[1] - a[1]) * axis[1]
                      + (b[2] - a[2]) * axis[2];
        vsum += (double)d * d;
      }
    }
    sum += vsum;
    count += (height - lag) * width;
  }

  return count ? sum / (double)count : 0.0;
}

// characteristic size of the image's color structure along the blind axis --
// measured, not assumed.
//
// for a field of variance s2, the mean squared difference between two points a
// distance h apart is 2*s2*(1 - rho(h)): zero at h = 0, rising to a 2*s2
// plateau once the two points are far enough apart to be uncorrelated. the
// distance at which it first reaches s2 is therefore where correlation has
// dropped to one half -- the "range" of the variogram, and a well defined
// length scale for how large this image's patches of color actually are.
//
// sampled on a geometric ladder and interpolated in log space. returns a length
// in pixels of the buffer it was measured on, or 0 if there was nothing to
// measure.
static double _axis_correlation_length(const float *const restrict in,
                                       const size_t width,
                                       const size_t height,
                                       const dt_aligned_pixel_t axis,
                                       const double variance)
{
  const size_t max_lag = MAX(1, MIN(width, height) / 4);

  double lags[32], gamma[32];
  int n = 0;
  for(size_t lag = 1; n < 32; lag *= 2)
  {
    const size_t l = MIN(lag, max_lag);
    lags[n] = (double)l;
    gamma[n] = _axis_variogram(in, width, height, axis, l);
    n++;
    if(l >= max_lag) break;
  }

  if(n < 2 || !(variance > 0.0)) return 0.0;

  // chroma noise is uncorrelated between neighbouring pixels, so it shows up as
  // a step the variogram already has at the shortest lags -- geostatistics
  // calls it the nugget. left in, it makes a noisy frame look as though its
  // color decorrelates within one pixel: measured on a sample of real raws it
  // reached 80% of the total on the worst of them, collapsing the estimate to
  // the minimum. extrapolate the first two lags back to h -> 0 and subtract it.
  const double nugget = fmax(2.0 * gamma[0] - gamma[1], 0.0);
  const double sill = 2.0 * variance - nugget;
  if(!(sill > 0.0)) return 0.0;

  // a quarter of the structured variance. the more usual half-decorrelation
  // point would be a longer, equally valid length, but on real photographs the
  // variogram frequently has not risen that far within the distances we can
  // sample -- their color keeps drifting out to frame scale -- and every such
  // image would fall through to the window edge, which measures our window
  // rather than the picture. a quarter is reached by essentially all of them.
  const double target = 0.25 * sill;

  for(int i = 0; i < n; i++)
  {
    const double g = fmax(gamma[i] - nugget, 0.0);
    if(g < target) continue;
    if(i == 0) return lags[0];

    const double prev = fmax(gamma[i - 1] - nugget, 0.0);
    const double t = (target - prev) / fmax(g - prev, 1e-12);
    return lags[i - 1] * pow(lags[i] / lags[i - 1], CLAMP(t, 0.0, 1.0));
  }

  // color structure coarser than anything we can measure here
  return lags[n - 1];
}

// analyze the image's per-pixel chrominance and pick the (filter) hue/chroma
// that maximizes tonal separation, plus a chroma_contrast setting for whatever
// separation no single global hue can capture.
//
// per pixel, project the RGB deviation from its own mean onto the same
// (cos, sin) basis the hue/chroma filter model uses for R/G/B at 0/120/240
// degrees, giving a 2D chrominance vector (x, y). with the filter switched on
// the output is exactly luminance + chroma * <(x, y), (cos hue, sin hue)>, so
// the hue that adds the most tonal variation is the leading eigenvector of the
// *covariance* matrix of the (x, y) distribution -- a small, classic PCA
// problem (it has to be the covariance, not the raw second moments: an overall
// color cast is a constant offset of (x, y) and carries no separation at all,
// yet would dominate uncentered moments and drag the axis towards itself).
//
// once the filter is settled, the mix it produces has a definite blind
// direction (see _blind_axis_hue()). the color variance that falls on *that*
// axis, relative to the dominant one, is what chroma_contrast has to work with,
// and sets its suggested strength.
//
// every quantity derived below is a ratio of same-unit values, so the result
// does not change if the image is scaled (i.e. exposed) differently.
static void _auto_compute(dt_iop_module_t *self,
                          dt_iop_blackwhite_gui_data_t *g,
                          const float *const restrict in,
                          const size_t width,
                          const size_t height,
                          const dt_iop_order_iccprofile_info_t *const work_profile)
{
  const size_t npixels = width * height;
  dt_aligned_pixel_t lum;
  _reference_luminance(lum);

  // the hue this produces is fed straight back into the filter, so the analysis
  // has to run in the space that hue will be read in -- the chrominance plane
  // below is spanned by the reference primaries, not by the pipe's.
  dt_colormatrix_t M;
  _work_to_reference(work_profile, M);

  const float cos120 = cosf(2.f * M_PI_F / 3.f);
  const float sin120 = sinf(2.f * M_PI_F / 3.f);
  const float cos240 = cosf(4.f * M_PI_F / 3.f);
  const float sin240 = sinf(4.f * M_PI_F / 3.f);

  double sl = 0.0, sx = 0.0, sy = 0.0;
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  double slx = 0.0, sly = 0.0;

  DT_OMP_FOR(reduction(+ : sl, sx, sy, sxx, syy, sxy, slx, sly))
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const px = in + 4 * k;
    dt_aligned_pixel_t rgb;
    for(int c = 0; c < 3; c++)
      rgb[c] = px[0] * M[c][0] + px[1] * M[c][1] + px[2] * M[c][2];

    const float lgt = rgb[0] * lum[0] + rgb[1] * lum[1] + rgb[2] * lum[2];
    const float mean = (rgb[0] + rgb[1] + rgb[2]) / 3.f;
    const float dR = rgb[0] - mean;
    const float dG = rgb[1] - mean;
    const float dB = rgb[2] - mean;

    const float x = dR + dG * cos120 + dB * cos240;
    const float y = dG * sin120 + dB * sin240;

    sl += lgt;
    sx += x;
    sy += y;
    sxx += (double)x * x;
    syy += (double)y * y;
    sxy += (double)x * y;
    slx += (double)lgt * x;
    sly += (double)lgt * y;
  }

  // auto_params was seeded with the current settings by _auto_callback(), so
  // anything not derived below (notably the detail radius, which is a matter of
  // taste and subject size rather than something measurable) is left alone.
  dt_iop_blackwhite_params_t *dp = &g->auto_params;
  dp->filter = FALSE;
  dp->hue = 0.f;
  dp->chroma = 0.f;
  dp->chroma_contrast = 0.f;
  // unlike the other three, this isn't derived from image content -- there's no
  // per-image "optimal" amount of human vision weighting, it's simply never
  // worse than a flat R/G/B average, so auto always turns it fully on.
  dp->eye_response = 1.f;

  if(npixels == 0) goto done;

  {
    const double n = (double)npixels;
    const double ml = sl / n, mx = sx / n, my = sy / n;
    // centered second moments: the covariance of the chrominance distribution
    const double cxx = fmax(sxx / n - mx * mx, 0.0);
    const double cyy = fmax(syy / n - my * my, 0.0);
    const double cxy = sxy / n - mx * my;
    // covariance of each chrominance axis with luminance, used to resolve the
    // eigenvector's sign below
    const double clx = slx / n - ml * mx;
    const double cly = sly / n - ml * my;

    const double avg = 0.5 * (cxx + cyy);
    const double spread = hypot(0.5 * (cxx - cyy), cxy);
    const double var_major = avg + spread;

    // the only scale reference we have: the image's own luminance level. all
    // comparisons are made against it so nothing depends on absolute exposure.
    const double lref = fmax(ml, 1e-6);

    // essentially neutral image: no hue can separate anything, leave the filter off
    if(var_major <= 1e-8 * lref * lref) goto done;

    double theta = 0.5 * atan2(2.0 * cxy, cxx - cyy);
    // an eigenvector is only defined up to a 180 degree flip, and the two
    // choices are *not* equivalent: the filtered result is luminance +
    // chroma * projection, so the sign that correlates positively with
    // luminance both maximizes the output's total variance and keeps the
    // natural tonal ordering (whatever was brighter stays brighter) instead of
    // inverting it.
    if(clx * cos(theta) + cly * sin(theta) < 0.0) theta += M_PI;

    double hue_deg = fmod(theta * 180.0 / M_PI, 360.0);
    if(hue_deg < 0.0) hue_deg += 360.0;

    const double sigma_major = sqrt(var_major);

    dp->filter = TRUE;
    dp->hue = (float)hue_deg;
    // the filter adds chroma * projection on top of the luminance. pick the
    // strength so that added swing stays a fixed fraction of the image's own
    // luminance level -- strong enough to separate, gentle enough not to push
    // large areas through the black clip. a strongly colored image therefore
    // needs less filter than a nearly neutral one. clamped to the range the
    // real Wratten filters above span.
    dp->chroma = (float)CLAMP(0.5 * lref / sigma_major, 0.15, 0.6);

    // now that the filter is decided, the mix it produces has a definite blind
    // direction; measure how much of the image's color variation actually falls
    // on it, relative to the dominant one, and suggest that as the strength.
    // zero when there is nothing the mix cannot already see.
    dt_iop_blackwhite_data_t mix = { .filter = dp->filter,
                                     .hue = dp->hue,
                                     .chroma = dp->chroma };
    const double phi = _blind_axis_hue(&mix);
    const double cp = cos(phi), sp = sin(phi);
    const double var_blind =
      fmax(cxx * cp * cp + 2.0 * cxy * cp * sp + cyy * sp * sp, 0.0);

    dp->chroma_contrast = (float)CLAMP(sqrt(var_blind) / sigma_major, 0.0, 1.0);

    // and how big that leftover color structure is, which is what the detail
    // radius has to match. only worth measuring if there is something there.
    if(var_blind > 1e-8 * lref * lref)
    {
      // measured on the pipe's own pixels, so the weights have to be carried
      // into its coordinates first. that projection is by construction the same
      // number the reference-space one would give, so it is directly comparable
      // to var_blind above.
      dt_aligned_pixel_t axis_ref, axis;
      _chroma_axis_weights((float)phi, axis_ref);
      _reference_to_work(M, axis_ref, axis);

      const double len = _axis_correlation_length(in, width, height, axis, var_blind);

      // both the measured length and the diagonal are in pixels of this same
      // preview, so their ratio is already the frame-relative quantity the
      // parameter wants -- no conversion through the pipe's scaling, and hence
      // no dependence on how large a preview we happened to be handed.
      //
      // the 1.5 is the one taste constant here: it sets how far above the
      // measured structure size the low-pass sits, and so whether whole patches
      // or only bands along their edges end up in the local term.
      if(len > 0.0)
      {
        const double diag = hypot((double)width, (double)height);
        dp->detail_radius = (float)CLAMP(1.5 * 100.0 * len / diag, 0.05, 10.0);
      }
    }
  }

done:
  dt_iop_gui_enter_critical_section(self);
  g->auto_state = 2;
  dt_iop_gui_leave_critical_section(self);
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4 /*we need full-color pixels*/, self, piece->colors,
                                        ivoid, ovoid, roi_in, roi_out))
    return;

  const dt_iop_blackwhite_data_t *const d = piece->data;
  const size_t width = roi_out->width;
  const size_t height = roi_out->height;
  const size_t npixels = width * height;
  const float *const restrict in = (const float *const)ivoid;
  float *const restrict out = (float *const)ovoid;

  const dt_iop_order_iccprofile_info_t *const work_profile =
    dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  dt_aligned_pixel_t grey;
  _compute_grey_mix(d, work_profile, grey);

  // the auto analysis needs to see the whole image at once, so skip it while
  // the pipe hands us one tile at a time -- the request simply stays pending
  // and is served by the next untiled preview run.
  if(self->dev->gui_attached
     && dt_pipe_is_preview(piece->pipe)
     && !piece->pipe->tiling)
  {
    dt_iop_blackwhite_gui_data_t *g = self->gui_data;
    if(g)
    {
      dt_iop_gui_enter_critical_section(self);
      const int auto_state = g->auto_state;
      dt_iop_gui_leave_critical_section(self);

      if(auto_state == 1)
        _auto_compute(self, g, in, width, height, work_profile);
    }
  }

  // chroma contrast: put back the color contrast the conversion is blind to.
  // _blind_axis_hue() gives the one chroma direction that maps to exactly zero
  // change in grey, so two regions differing only along it collapse onto the
  // same shade no matter how the mix is set up.
  //
  // only the local part of that projection is added. its low-frequency part is
  // not new information -- adding it globally is arithmetically the same as
  // changing the mix weights, which is what the chroma and human vision sliders
  // already do -- so the radius is precisely the boundary between what the
  // global mix handles and what is left for the local term.
  //
  // this is a cheap, tileable stand-in for the goal of the Color2Gray/Decolorize
  // family rather than an implementation of either -- those need a global solve
  // or a considerably more elaborate perceptual pipeline.
  float *detail = NULL;
  gboolean local_pass = d->chroma_contrast > 0.f;
  if(local_pass
     && !dt_iop_alloc_image_buffers(self, roi_in, roi_out, 1, &detail, 0, NULL))
  {
    // out of memory: still deliver a correct black & white image, just without
    // the local refinement
    local_pass = FALSE;
  }

  if(!local_pass)
  {
    DT_OMP_FOR()
    for(size_t k = 0; k < npixels; k++)
    {
      const float *const px = in + 4 * k;
      float *const o = out + 4 * k;
      const float grey_mix = fmaxf(px[0] * grey[0] + px[1] * grey[1] + px[2] * grey[2], 0.f);
      o[0] = o[1] = o[2] = grey_mix;
      o[3] = px[3];
    }
    return;
  }

  dt_aligned_pixel_t axis_ref, axis;
  _chroma_axis_weights(_blind_axis_hue(d), axis_ref);
  dt_colormatrix_t M;
  _work_to_reference(work_profile, M);
  _reference_to_work(M, axis_ref, axis);

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const px = in + 4 * k;
    detail[k] = px[0] * axis[0] + px[1] * axis[1] + px[2] * axis[2];
  }

  // low-pass in place: 'detail' now holds the part of that projection which is
  // redundant with the hue choice, and gets subtracted below
  dt_box_mean(detail, height, width, 1, _chroma_contrast_radius(piece, roi_in), BOX_ITERATIONS);

  const float strength = d->chroma_contrast * BLACKWHITE_CHROMA_CONTRAST_GAIN;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const px = in + 4 * k;
    const float grey_mix = px[0] * grey[0] + px[1] * grey[1] + px[2] * grey[2];
    const float projection = px[0] * axis[0] + px[1] * axis[1] + px[2] * axis[2];

    float *const o = out + 4 * k;
    const float value = fmaxf(grey_mix + strength * (projection - detail[k]), 0.f);
    o[0] = o[1] = o[2] = value;
    o[3] = px[3];
  }

  dt_free_align(detail);
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  const dt_iop_blackwhite_data_t *const d = piece->data;
  dt_iop_blackwhite_global_data_t *gd = self->global_data;

  // commit_params() already clears process_cl_ready in that case; belt and
  // braces, since only the CPU path implements the local neighborhood pass
  if(d->chroma_contrast > 0.f)
    return DT_OPENCL_PROCESS_CL;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const dt_iop_order_iccprofile_info_t *const work_profile =
    dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  dt_aligned_pixel_t grey;
  _compute_grey_mix(d, work_profile, grey);

  return dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_blackwhite, width, height,
    CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(grey));
}
#endif

void init_global(dt_iop_module_so_t *self)
{
  const int program = 8; // extended.cl, from programs.conf
  dt_iop_blackwhite_global_data_t *gd = malloc(sizeof(dt_iop_blackwhite_global_data_t));
  self->data = gd;
  gd->kernel_blackwhite = dt_opencl_create_kernel(program, "blackwhite");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_blackwhite_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_blackwhite);
  free(self->data);
  self->data = NULL;
}

// same shape as _compute_grey_mix()'s filter term, but clamped to a displayable
// [0, 1] range instead of normalized to sum=1 -- so the swatch reads as neutral
// grey at chroma=0 and grows more saturated as chroma increases, the way an
// actual colored filter would look. the model is anchored to Rec.709 primaries
// and these are drawn as screen RGB, so the swatch is the filter's own color
// rather than an approximation of it.
static void _hue_chroma_to_rgb(const float hue_deg, const float chroma, dt_aligned_pixel_t RGB)
{
  const float hue = deg2radf(hue_deg);
  for(int c = 0; c < 3; c++)
    RGB[c] = CLAMP(1.f / 3.f + chroma * cosf(hue - c * (2.f * M_PI_F / 3.f)), 0.f, 1.f);
}

static gboolean _filter_color_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  const dt_iop_blackwhite_params_t *p = self->params;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width;
  int height = allocation.height;
  cairo_surface_t *cst = dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);

  const double INNER_PADDING = 4.0;
  const float margin = 2. * DT_PIXEL_APPLY_DPI(1.5);
  width -= 2 * INNER_PADDING;
  height -= 2 * margin;

  dt_aligned_pixel_t RGB = { 0 };
  _hue_chroma_to_rgb(p->hue, p->chroma, RGB);
  cairo_set_source_rgb(cr, RGB[0], RGB[1], RGB[2]);
  cairo_rectangle(cr, INNER_PADDING, margin, width, height);
  cairo_fill(cr);

  cairo_stroke(cr);
  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return TRUE;
}

static void _preview_pipe_finished_callback(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_blackwhite_gui_data_t *g = self->gui_data;
  if(!g) return;

  dt_iop_gui_enter_critical_section(self);
  const int state = g->auto_state;
  dt_iop_gui_leave_critical_section(self);
  if(state != 2) return;

  dt_iop_gui_enter_critical_section(self);
  const dt_iop_blackwhite_params_t dp = g->auto_params;
  g->auto_state = 0;
  dt_iop_gui_leave_critical_section(self);

  // write directly into params and commit *before* refreshing the widgets:
  // a bound bauhaus widget set while DT_ENTER/LEAVE_GUI_UPDATE is active only
  // moves on screen, it deliberately skips writing the value back into params
  // (that's the guard's whole point, it normally runs the other way around --
  // syncing widgets to params that already changed). so the refresh below is
  // purely cosmetic; the param write and the history commit have to happen here.
  dt_iop_blackwhite_params_t *p = self->params;
  *p = dp;

  dt_dev_add_history_item(darktable.develop, self, TRUE);

  dt_iop_gui_update(self); // pulls every bound widget back from params
  gui_changed(self, NULL, NULL); // refresh swatch visibility/redraw for the new filter state
}

static void _auto_callback(GtkButton *button, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_blackwhite_gui_data_t *g = self->gui_data;

  dt_iop_request_focus(self);
  if(self->off && !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(self->off)))
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  dt_iop_gui_enter_critical_section(self);
  // seed with the current settings so the analysis only has to overwrite what
  // it actually derives, and leaves the rest untouched
  g->auto_params = *(const dt_iop_blackwhite_params_t *)self->params;
  g->auto_state = 1;
  dt_iop_gui_leave_critical_section(self);

  dt_dev_reprocess_preview(self->dev, self->iop_order);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_blackwhite_params_t *p = self->params;
  dt_iop_blackwhite_gui_data_t *g = self->gui_data;

  if(!w || w == g->filter)
  {
    gtk_widget_set_visible(g->swatch, p->filter);
    gtk_widget_set_visible(g->hue, p->filter);
    gtk_widget_set_visible(g->chroma, p->filter);
  }

  if(!w || w == g->hue || w == g->chroma)
    gtk_widget_queue_draw(g->swatch);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_blackwhite_gui_data_t *g = IOP_GUI_ALLOC(blackwhite);

  g->auto_state = 0;

  // self->widget is lazily created as a vbox by the first dt_bauhaus_*_from_params
  // call; the auto button is added before any of those, so it has to make sure
  // that's happened first instead of handing dt_gui_box_add a NULL container.
  if(!self->widget) self->widget = dt_gui_vbox();

  g->auto_button = dt_action_button_new
    (NULL, N_("auto"), _auto_callback, self,
     _("analyze the image: pick the filter hue and chroma that maximize tonal\n"
       "separation, measure how much color contrast is left over and at what\n"
       "size, and switch on human vision weighting"), 0, 0);
  gtk_widget_set_size_request(g->auto_button, -1, DT_PIXEL_APPLY_DPI(24));
  dt_gui_box_add(self->widget, g->auto_button);

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished_callback);

  g->eye_response = dt_bauhaus_slider_from_params(self, "eye_response");
  dt_bauhaus_slider_set_soft_range(g->eye_response, 0.0, 1.0);
  gtk_widget_set_tooltip_text
    (g->eye_response,
     _("blend the base panchromatic mix toward the eye's actual color sensitivity\n"
       "(the human eye is far more sensitive to green than to red or blue) instead\n"
       "of a flat, equal-parts average of red, green and blue.\n"
       "any color filter below is applied on top of this."));

  g->filter = dt_bauhaus_toggle_from_params(self, "filter");
  gtk_widget_set_tooltip_text
    (g->filter,
     _("simulate a colored lens filter:\n"
       "lightens tones close to the filter hue, darkens tones far from it,\n"
       "the way red/orange/yellow/green filters work on black & white film."));

  g->swatch = GTK_WIDGET(gtk_drawing_area_new());
  gtk_widget_set_size_request
    (g->swatch, 2 * DT_PIXEL_APPLY_DPI(darktable.bauhaus->quad_width),
     DT_PIXEL_APPLY_DPI(darktable.bauhaus->quad_width));
  gtk_widget_set_tooltip_text(g->swatch, _("preview of the current filter color"));
  g_signal_connect(G_OBJECT(g->swatch), "draw", G_CALLBACK(_filter_color_draw), self);
  dt_gui_box_add(self->widget, g->swatch);

  g->hue = dt_bauhaus_slider_from_params(self, "hue");
  dt_bauhaus_slider_set_format(g->hue, "°");
  gtk_widget_set_tooltip_text
    (g->hue,
     _("hue of the virtual color filter.\n"
       "it also selects the color axis used by chroma contrast below,\n"
       "which works on the colors perpendicular to this hue."));
  for(int i = 0; i <= 6; i++)
  {
    const float stop = i / 6.0f;
    dt_aligned_pixel_t rgb = { 0 };
    _hue_chroma_to_rgb(stop * 360.f, 2.f / 3.f, rgb);
    dt_bauhaus_slider_set_stop(g->hue, stop, rgb[0], rgb[1], rgb[2]);
  }

  g->chroma = dt_bauhaus_slider_from_params(self, "chroma");
  dt_bauhaus_slider_set_soft_range(g->chroma, 0.0, 1.0);
  gtk_widget_set_tooltip_text(g->chroma, _("strength of the virtual color filter"));

  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "chroma contrast")));

  g->chroma_contrast = dt_bauhaus_slider_from_params(self, "chroma_contrast");
  dt_bauhaus_slider_set_soft_range(g->chroma_contrast, 0.0, 1.0);
  gtk_widget_set_tooltip_text
    (g->chroma_contrast,
     _("restore local contrast between colors that any black & white conversion\n"
       "renders as the same shade -- keeps differently colored subjects of equal\n"
       "lightness from blending together.\n"
       "the colors it acts on follow from the conversion itself, so this works\n"
       "with or without a color filter."));

  g->detail_radius = dt_bauhaus_slider_from_params(self, "detail_radius");
  // the whole hard range is usable: 8 box passes turn a radius of 10% of the
  // diagonal into a sigma of about a third of the frame height, which is as
  // close to a global mean as this is worth taking. auto reaches that ceiling
  // on any image whose color only decorrelates at frame scale, so dragging has
  // to be able to get there too. only the very bottom is trimmed.
  dt_bauhaus_slider_set_soft_range(g->detail_radius, 0.1, 10.0);
  dt_bauhaus_slider_set_format(g->detail_radius, " %");
  // the parameter already is a percentage, but a "%" format on a slider whose
  // hard max is <= 10 makes bauhaus assume a 0..1 fraction and rescale by 100
  // (and drop two digits), so undo both after setting the format.
  dt_bauhaus_slider_set_factor(g->detail_radius, 1.0f);
  dt_bauhaus_slider_set_digits(g->detail_radius, 2);
  gtk_widget_set_tooltip_text
    (g->detail_radius,
     _("size of the detail chroma contrast works on, as a percentage of the\n"
       "image diagonal. anything coarser than this is left to the sliders\n"
       "above: small values separate fine texture, large values whole subjects.\n"
       "auto measures how large this image's patches of color actually are."));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
