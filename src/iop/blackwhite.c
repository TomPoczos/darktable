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
#include "common/color_picker.h"
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
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "libs/lib.h"

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

// which settings a single area measurement is allowed to write. one flag per
// area picker, so each picker only ever touches what it is labelled to set and
// leaves the user's other choices alone.
typedef enum _bw_auto_mode_t
{
  BW_AUTO_NONE            = 0,
  BW_AUTO_FILTER          = 1 << 0, // filter on/off, hue, chroma
  BW_AUTO_CHROMA_CONTRAST = 1 << 1, // chroma contrast strength and detail radius
  BW_AUTO_EYE_RESPONSE    = 1 << 2, // human vision weighting
  BW_AUTO_ALL = BW_AUTO_FILTER | BW_AUTO_CHROMA_CONTRAST | BW_AUTO_EYE_RESPONSE
} _bw_auto_mode_t;

typedef struct dt_iop_blackwhite_gui_data_t
{
  GtkWidget *filter;          // also carries the color filter area picker in its quad
  GtkWidget *swatch;
  GtkWidget *hue;
  GtkWidget *chroma;
  GtkWidget *chroma_contrast; // also carries the chroma contrast area picker
  GtkWidget *detail_radius;
  GtkWidget *eye_response;
  GtkWidget *picker_all;      // standalone picker driving all of the above at once

  // cross-thread hand-off for the area pickers: color_picker_apply() records
  // what the picked area is meant to drive and flips auto_state to 1;
  // process() measures on a copy when it sees state 1 on the preview pipe and
  // writes the result back as state 2, but only if auto_request still names
  // the pick it answered, so a drag made meanwhile is never overwritten by a
  // stale result; the preview-pipe-finished signal handler (GUI thread) then
  // applies auto_params and resets state to 0. every access is under
  // dt_iop_gui_enter/leave_critical_section.
  int auto_state; // 0: idle, 1: computation requested, 2: result ready to apply
  unsigned auto_request; // bumped by every pick
  _bw_auto_mode_t auto_mode;
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

// the rectangle an area picker asked us to measure, and the buffer it sits in.
//
// the frame dimensions are carried along separately from the region's own
// because the detail radius is a percentage of the *image* diagonal, not of
// whatever the user happened to draw a box around: a patch of colour a
// fiftieth of the frame across has to come out as 2% whether it was measured
// on the whole picture or on one corner of it.
typedef struct _bw_region_t
{
  const float *in;          // start of the buffer, 4 floats per pixel
  size_t stride;            // buffer width, in pixels
  size_t x0, y0;            // region origin within the buffer
  size_t width, height;     // region size
  size_t frame_w, frame_h;  // buffer size, i.e. the whole frame
} _bw_region_t;

// mean squared difference of the blind-axis projection between pairs of pixels
// a given distance apart, averaged over the horizontal and vertical directions.
// note the projection weights sum to zero, so constant offsets drop out and
// this measures purely how much the *color* changes over that distance.
static double _axis_variogram(const _bw_region_t *const region,
                              const dt_aligned_pixel_t axis,
                              const size_t lag)
{
  const float *const restrict in = region->in;
  const size_t stride = region->stride;
  const size_t x0 = region->x0, y0 = region->y0;
  const size_t width = region->width, height = region->height;

  double sum = 0.0;
  size_t count = 0;

  if(width > lag)
  {
    double hsum = 0.0;
    DT_OMP_FOR(reduction(+ : hsum))
    for(size_t j = 0; j < height; j++)
    {
      const float *const row = in + 4 * ((y0 + j) * stride + x0);
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
      const float *const ra = in + 4 * ((y0 + j) * stride + x0);
      const float *const rb = in + 4 * ((y0 + j + lag) * stride + x0);
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
    count += rows * width;
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
static double _axis_correlation_length(const _bw_region_t *const region,
                                       const dt_aligned_pixel_t axis,
                                       const double variance)
{
  // the ladder is bounded by the *region*, not the frame: a small picked area
  // simply cannot show that color keeps decorrelating beyond its own edge, so
  // it saturates sooner and reports a smaller structure. picking large is what
  // makes a large answer available.
  const size_t max_lag = MAX(1, MIN(region->width, region->height) / 4);

  double lags[32], gamma[32];
  int n = 0;
  for(size_t lag = 1; n < 32; lag *= 2)
  {
    const size_t l = MIN(lag, max_lag);
    lags[n] = (double)l;
    gamma[n] = _axis_variogram(region, axis, l);
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

// the second-moment statistics every area measurement below is built from.
//
// per pixel, project the RGB deviation from its own mean onto the same
// (cos, sin) basis the hue/chroma filter model uses for R/G/B at 0/120/240
// degrees, giving a 2D chrominance vector (x, y), and accumulate the
// *covariance* of that distribution -- it has to be the covariance, not the
// raw second moments: an overall color cast is a constant offset of (x, y) and
// carries no separation at all, yet would dominate uncentered moments and drag
// the axis towards itself.
//
// every quantity derived from these is a ratio of same-unit values, so nothing
// below changes if the image is scaled (i.e. exposed) differently.
typedef struct _bw_moments_t
{
  double lref;           // the region's mean luminance: our only scale reference
  double cxx, cyy, cxy;  // covariance of the chrominance distribution
  double var_major;      // its leading eigenvalue
  double theta;          // direction of the matching eigenvector, in radians
  gboolean valid;        // enough pixels, and some color for a hue to act on
} _bw_moments_t;

static void _region_moments(const _bw_region_t *const region,
                            const dt_colormatrix_t M,
                            _bw_moments_t *const m)
{
  memset(m, 0, sizeof(*m));

  const float *const restrict in = region->in;
  const size_t stride = region->stride;
  const size_t x0 = region->x0, y0 = region->y0;
  const size_t width = region->width, height = region->height;
  const size_t npixels = width * height;
  if(npixels == 0) return;

  dt_aligned_pixel_t lum;
  _reference_luminance(lum);

  const float cos120 = cosf(2.f * M_PI_F / 3.f);
  const float sin120 = sinf(2.f * M_PI_F / 3.f);
  const float cos240 = cosf(4.f * M_PI_F / 3.f);
  const float sin240 = sinf(4.f * M_PI_F / 3.f);

  double sl = 0.0, sx = 0.0, sy = 0.0;
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  double slx = 0.0, sly = 0.0;

  DT_OMP_FOR(reduction(+ : sl, sx, sy, sxx, syy, sxy, slx, sly))
  for(size_t j = 0; j < height; j++)
  {
    const float *const row = in + 4 * ((y0 + j) * stride + x0);
    for(size_t i = 0; i < width; i++)
    {
      const float *const px = row + 4 * i;
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
  }

  const double n = (double)npixels;
  const double ml = sl / n, mx = sx / n, my = sy / n;
  // centered second moments: the covariance of the chrominance distribution
  m->cxx = fmax(sxx / n - mx * mx, 0.0);
  m->cyy = fmax(syy / n - my * my, 0.0);
  m->cxy = sxy / n - mx * my;
  // covariance of each chrominance axis with luminance, used to resolve the
  // eigenvector's sign below
  const double clx = slx / n - ml * mx;
  const double cly = sly / n - ml * my;

  const double avg = 0.5 * (m->cxx + m->cyy);
  const double spread = hypot(0.5 * (m->cxx - m->cyy), m->cxy);
  m->var_major = avg + spread;

  // the only scale reference we have: the region's own luminance level. all
  // comparisons are made against it so nothing depends on absolute exposure.
  m->lref = fmax(ml, 1e-6);

  // essentially neutral area: no hue can separate anything in it
  if(m->var_major <= 1e-8 * m->lref * m->lref) return;

  double theta = 0.5 * atan2(2.0 * m->cxy, m->cxx - m->cyy);
  // an eigenvector is only defined up to a 180 degree flip, and the two
  // choices are *not* equivalent: the filtered result is luminance +
  // chroma * projection, so the sign that correlates positively with
  // luminance both maximizes the output's total variance and keeps the
  // natural tonal ordering (whatever was brighter stays brighter) instead of
  // inverting it.
  if(clx * cos(theta) + cly * sin(theta) < 0.0) theta += M_PI;

  m->theta = theta;
  m->valid = TRUE;
}

// pick the (filter) hue/chroma that maximizes tonal separation over the region.
//
// with the filter switched on the output is exactly luminance + chroma *
// <(x, y), (cos hue, sin hue)>, so the hue that adds the most tonal variation
// is the leading eigenvector of the covariance matrix of the (x, y)
// distribution -- a small, classic PCA problem, already solved in
// _region_moments().
static void _auto_filter(dt_iop_blackwhite_params_t *const dp,
                         const _bw_moments_t *const m)
{
  dp->filter = FALSE;
  dp->hue = 0.f;
  dp->chroma = 0.f;

  // nothing to separate: leave the filter off
  if(!m->valid) return;

  double hue_deg = fmod(m->theta * 180.0 / M_PI, 360.0);
  if(hue_deg < 0.0) hue_deg += 360.0;

  dp->filter = TRUE;
  dp->hue = (float)hue_deg;
  // the filter adds chroma * projection on top of the luminance. pick the
  // strength so that added swing stays a fixed fraction of the area's own
  // luminance level -- strong enough to separate, gentle enough not to push
  // large areas through the black clip. a strongly colored subject therefore
  // needs less filter than a nearly neutral one. clamped to the range the
  // real Wratten filters above span.
  dp->chroma = (float)CLAMP(0.5 * m->lref / sqrt(m->var_major), 0.15, 0.6);
}

// with the filter settled, the mix it produces has a definite blind direction
// (see _blind_axis_hue()). the color variance that falls on *that* axis,
// relative to the dominant one, is what chroma contrast has to work with and
// sets its suggested strength; how large that leftover structure is sets the
// detail radius.
//
// the filter is read straight out of dp, so measuring an area for chroma
// contrast on its own answers the question for whatever mix the module is
// currently set to -- picked, preset or dialed in by hand.
static void _auto_chroma_contrast(dt_iop_blackwhite_params_t *const dp,
                                  const _bw_moments_t *const m,
                                  const _bw_region_t *const region,
                                  const dt_colormatrix_t M)
{
  // zero when there is nothing the mix cannot already see
  dp->chroma_contrast = 0.f;

  if(!m->valid) return;

  const dt_iop_blackwhite_data_t mix = { .filter = dp->filter,
                                         .hue = dp->hue,
                                         .chroma = dp->chroma };
  const double phi = _blind_axis_hue(&mix);
  const double cp = cos(phi), sp = sin(phi);
  const double var_blind =
    fmax(m->cxx * cp * cp + 2.0 * m->cxy * cp * sp + m->cyy * sp * sp, 0.0);

  dp->chroma_contrast =
    (float)CLAMP(sqrt(var_blind) / sqrt(m->var_major), 0.0, 1.0);

  // and how big that leftover color structure is, which is what the detail
  // radius has to match. only worth measuring if there is something there.
  if(var_blind <= 1e-8 * m->lref * m->lref) return;

  // measured on the pipe's own pixels, so the weights have to be carried
  // into its coordinates first. that projection is by construction the same
  // number the reference-space one would give, so it is directly comparable
  // to var_blind above.
  dt_aligned_pixel_t axis_ref, axis;
  _chroma_axis_weights((float)phi, axis_ref);
  _reference_to_work(M, axis_ref, axis);

  const double len = _axis_correlation_length(region, axis, var_blind);
  if(!(len > 0.0)) return;

  // both the measured length and the diagonal are in pixels of this same
  // preview, so their ratio is already the frame-relative quantity the
  // parameter wants -- no conversion through the pipe's scaling, and hence
  // no dependence on how large a preview we happened to be handed.
  //
  // the diagonal is the *frame's*, never the picked region's: the parameter
  // means "this fraction of the image", so a structure of a given size has to
  // yield the same number however small an area it was measured in.
  //
  // the 1.5 is the one taste constant here: it sets how far above the
  // measured structure size the low-pass sits, and so whether whole patches
  // or only bands along their edges end up in the local term.
  const double diag = hypot((double)region->frame_w, (double)region->frame_h);
  dp->detail_radius = (float)CLAMP(1.5 * 100.0 * len / diag, 0.05, 10.0);
}

// analyze the area one of the pickers was dragged over and write the settings
// it is responsible for into dp.
//
// mode decides what gets written: the picker on the color filter toggle sets
// the filter alone, the one on the chroma contrast slider sets that pair
// alone, and the combined picker at the top of the module sets everything.
// whatever is not in mode keeps the value color_picker_apply() seeded dp
// with, i.e. is left exactly as the user had it.
static void _auto_compute(dt_iop_blackwhite_params_t *const dp,
                          const _bw_region_t *const region,
                          const _bw_auto_mode_t mode,
                          const dt_iop_order_iccprofile_info_t *const work_profile)
{
  // the hue this produces is fed straight back into the filter, so the analysis
  // has to run in the space that hue will be read in -- the chrominance plane
  // it works on is spanned by the reference primaries, not by the pipe's.
  dt_colormatrix_t M;
  _work_to_reference(work_profile, M);

  _bw_moments_t m;
  _region_moments(region, M, &m);

  // unlike the rest this isn't derived from image content -- there's no
  // per-image "optimal" amount of human vision weighting, it's simply never
  // worse than a flat R/G/B average, so the combined picker always turns it
  // fully on.
  if(mode & BW_AUTO_EYE_RESPONSE) dp->eye_response = 1.f;
  // order matters: the blind axis chroma contrast works on follows from the
  // filter, so a combined pick has to settle the filter first.
  if(mode & BW_AUTO_FILTER) _auto_filter(dp, &m);
  if(mode & BW_AUTO_CHROMA_CONTRAST) _auto_chroma_contrast(dp, &m, region, M);
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

  // an area measurement needs to see a whole, contiguous frame, so skip it
  // while the pipe hands us one tile at a time -- the request simply stays
  // pending and is served by the next untiled preview run.
  if(self->dev->gui_attached
     && dt_pipe_is_preview(piece->pipe)
     && !piece->pipe->tiling)
  {
    dt_iop_blackwhite_gui_data_t *g = self->gui_data;
    if(g)
    {
      dt_iop_gui_enter_critical_section(self);
      const int auto_state = g->auto_state;
      const unsigned auto_request = g->auto_request;
      const _bw_auto_mode_t auto_mode = g->auto_mode;
      dt_iop_blackwhite_params_t auto_params = g->auto_params;
      dt_iop_gui_leave_critical_section(self);

      if(auto_state == 1)
      {
        // the picker hands us a mean over the area, which is nowhere near
        // enough here (we need a covariance and a variogram over the actual
        // pixels), so take the box it drew and do the measuring ourselves.
        // the region defaults to the whole frame, which is also what we fall
        // back to if the box cannot be mapped into this module's coordinates.
        _bw_region_t region = { .in = in,
                                .stride = roi_in->width,
                                .x0 = 0, .y0 = 0,
                                .width = roi_in->width, .height = roi_in->height,
                                .frame_w = roi_in->width, .frame_h = roi_in->height };

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

        _auto_compute(&auto_params, &region, auto_mode, work_profile);

        // a newer pick has re-seeded auto_params and asked for a preview run
        // of its own, and losing focus withdraws the request altogether; in
        // both cases this result is for nobody
        dt_iop_gui_enter_critical_section(self);
        if(g->auto_state == 1 && g->auto_request == auto_request)
        {
          g->auto_params = auto_params;
          g->auto_state = 2;
        }
        dt_iop_gui_leave_critical_section(self);
      }
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
  // the refresh below deliberately runs under the gui-update guard and so
  // never writes anything back, which is the whole point of that guard -- it
  // normally runs the other way around, syncing widgets to params that have
  // already changed. the param write and the history commit have to happen here.
  dt_iop_blackwhite_params_t *p = self->params;
  *p = dp;

  dt_dev_add_history_item(darktable.develop, self, TRUE);

  // pull every bound widget back from params by hand rather than through
  // dt_iop_gui_update(): that also refreshes the blending UI, which switches
  // the color picker off outright (keep = FALSE) whenever blending is not in
  // parametric mode -- and the picker that asked for this measurement is
  // meant to stay armed so the area can be dragged again.
  //
  // the guard is what makes this refresh purely cosmetic: a bound bauhaus
  // widget set while DT_ENTER/LEAVE_GUI_UPDATE is active only moves on
  // screen, it deliberately skips writing the value back into params and
  // committing history -- which we have already done above, and which would
  // reset the picker here too.
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_update_from_field(self, NULL, NULL, NULL);
  DT_LEAVE_GUI_UPDATE();

  gui_changed(self, NULL, NULL); // refresh swatch visibility/redraw for the new filter state
}

// all three area pickers land here once the pipe has sampled their box. we
// ignore the sampled color itself -- what we want is the pixels underneath it,
// which only process() can see -- so this just records what the pick is meant
// to set and asks for one more preview pass to do the measuring on.
//
// the picker machinery only calls us when the box has actually moved, so
// applying the result (which commits history and re-runs the pipe) cannot
// bounce straight back in here: the analysis runs exactly once per drag.
void color_picker_apply(dt_iop_module_t *self,
                        GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_blackwhite_gui_data_t *g = self->gui_data;
  if(!g) return;

  _bw_auto_mode_t mode = BW_AUTO_NONE;
  if(picker == g->filter)               mode = BW_AUTO_FILTER;
  else if(picker == g->chroma_contrast) mode = BW_AUTO_CHROMA_CONTRAST;
  else if(picker == g->picker_all)      mode = BW_AUTO_ALL;
  else return;

  dt_iop_gui_enter_critical_section(self);
  // seed with the current settings so the measurement only has to overwrite
  // what this particular picker derives, and leaves the rest untouched
  g->auto_params = *(const dt_iop_blackwhite_params_t *)self->params;
  g->auto_mode = mode;
  g->auto_request++;
  g->auto_state = 1;
  dt_iop_gui_leave_critical_section(self);

  dt_dev_reprocess_preview(self->dev, self->iop_order);
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(in) return;

  dt_iop_color_picker_reset(self, TRUE);

  // the picker that asked for a measurement has just been switched off, and
  // on an image switch the result would land in the wrong image's history
  dt_iop_blackwhite_gui_data_t *g = self->gui_data;
  dt_iop_gui_enter_critical_section(self);
  g->auto_state = 0;
  dt_iop_gui_leave_critical_section(self);
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

// the bound widgets are synced by the framework before this is called; what
// it has to do is bring the filter section's visibility and the swatch in
// line with them, which nothing else does when params change under the GUI
void gui_update(dt_iop_module_t *self)
{
  gui_changed(self, NULL, NULL);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_blackwhite_gui_data_t *g = IOP_GUI_ALLOC(blackwhite);

  g->auto_state = 0;
  g->auto_request = 0;
  g->auto_mode = BW_AUTO_NONE;

  // self->widget is lazily created as a vbox by the first dt_bauhaus_*_from_params
  // call; the combined picker is added before any of those, so it has to make
  // sure that's happened first instead of handing dt_gui_box_add a NULL container.
  if(!self->widget) self->widget = dt_gui_vbox();

  // the "do all of it" entry point, first in the module because it is where a
  // conversion is meant to start. the two pickers further down then let you
  // re-answer either half of it from a different part of the picture without
  // disturbing the other -- the same shape as agx's per-slider pickers next to
  // its combined "auto tune levels".
  GtkWidget *auto_box = dt_gui_hbox();
  g->picker_all = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, NULL);
  gtk_widget_set_tooltip_text
    (g->picker_all,
     _("set up the whole conversion from an area of the image: pick the filter\n"
       "hue and chroma that maximize tonal separation there, measure how much\n"
       "color contrast is left over and at what size, and switch on human\n"
       "vision weighting.\n"
       "click to use the whole frame, then drag on the image to work from the\n"
       "subject that matters instead."));
  dt_action_define_iop(self, NULL, N_("auto from area"),
                       g->picker_all, &dt_action_def_color_picker);
  dt_gui_box_add(auto_box, dt_gui_expand(dt_ui_label_new(_("auto from area"))),
                 g->picker_all);
  dt_gui_box_add(self->widget, auto_box);

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished_callback);

  g->eye_response = dt_bauhaus_slider_from_params(self, "eye_response");
  dt_bauhaus_slider_set_soft_range(g->eye_response, 0.0, 1.0);
  gtk_widget_set_tooltip_text
    (g->eye_response,
     _("blend the base panchromatic mix toward the eye's actual color sensitivity\n"
       "(the human eye is far more sensitive to green than to red or blue) instead\n"
       "of a flat, equal-parts average of red, green and blue.\n"
       "any color filter below is applied on top of this."));

  // the picker sits on the toggle rather than on the hue slider because it
  // sets hue *and* chroma, and because the toggle is the one widget of the
  // three that is visible even when the filter is off -- so picking an area
  // can switch the filter on, which is half of what it is for.
  g->filter = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                  dt_bauhaus_toggle_from_params(self, "filter"));
  gtk_widget_set_tooltip_text
    (g->filter,
     _("simulate a colored lens filter:\n"
       "lightens tones close to the filter hue, darkens tones far from it,\n"
       "the way red/orange/yellow/green filters work on black & white film."));
  dt_bauhaus_widget_set_quad_tooltip
    (g->filter,
     _("pick an area: switch the filter on and set the hue and chroma that\n"
       "separate that area's colors most strongly"));

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

  g->chroma_contrast = dt_color_picker_new
    (self, DT_COLOR_PICKER_AREA,
     dt_bauhaus_slider_from_params(self, "chroma_contrast"));
  dt_bauhaus_slider_set_soft_range(g->chroma_contrast, 0.0, 1.0);
  gtk_widget_set_tooltip_text
    (g->chroma_contrast,
     _("restore local contrast between colors that any black & white conversion\n"
       "renders as the same shade -- keeps differently colored subjects of equal\n"
       "lightness from blending together.\n"
       "the colors it acts on follow from the conversion itself, so this works\n"
       "with or without a color filter."));
  // one picker for the pair below it: the strength and the size are two
  // readings of the same measurement and are meaningless apart.
  dt_bauhaus_widget_set_quad_tooltip
    (g->chroma_contrast,
     _("pick an area: measure how much color contrast the current mix cannot\n"
       "see there, and how large it is, and set the strength and detail radius\n"
       "from it"));

  g->detail_radius = dt_bauhaus_slider_from_params(self, "detail_radius");
  // the whole hard range is usable: 8 box passes turn a radius of 10% of the
  // diagonal into a sigma of about a third of the frame height, which is as
  // close to a global mean as this is worth taking. a pick over an image whose
  // color only decorrelates at frame scale reaches that ceiling, so dragging
  // has to be able to get there too. only the very bottom is trimmed.
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
       "the picker above measures how large the patches of color in the area\n"
       "you drag over actually are, and expresses that as a fraction of the\n"
       "whole frame -- so pick over a subject whose color structure is the one\n"
       "you want separated."));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
