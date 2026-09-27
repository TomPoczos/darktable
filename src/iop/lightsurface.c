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

// there is no process_cl(): the per-pixel work is a single gain, and the
// decomposition runs on ONNX Runtime whichever path the pipe takes. with
// OpenCL on, the pixelpipe sees no process_cl, copies this module's input
// back to host memory and runs process() on the CPU (possible_cl in
// src/develop/pixelpipe_hb.c)

#include "bauhaus/bauhaus.h"
#include "common/imagebuf.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#ifdef HAVE_AI
#include "common/ai/intrinsic.h"
#include "common/box_filters.h"
#include "common/chromatic_adaptation.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/guided_filter.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/pfm.h"
#include "control/control.h"
#include "control/signal.h"
#include "develop/develop.h"
#include "develop/pixelpipe_cache.h"
#include "develop/pixelpipe_hb.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#endif

#include <gtk/gtk.h>
#include <string.h>

DT_MODULE_INTROSPECTION(1, dt_iop_lightsurface_params_t)

// light curve nodes sit at x_i = -7 + i EV; surface hue nodes at 45 degree steps
#define LS_LIGHT_NODES 9
#define LS_HUE_NODES 8

// minimum output rise between adjacent light nodes, in EV
#define LS_NODE_MIN_STEP 0.1f

// saturation at which a surface hue edit reaches full weight (epic §M4.2);
// 0.6 per the LS-03 decision record, where it moved near-neutral pixels least
#define LS_SAT_FULL 0.6f
// light level, in EV above the key light, that the warning overlay flags
#define LS_WARN_LAMBDA 1.0f
// the log guide's floor, in EV below the median luminance (epic §M1.7)
#define LS_GUIDE_FLOOR_EV 10.0f
// percentile of log shading that defines the key light (epic §M1.3)
#define LS_KEY_PERCENTILE 90.0
// the floor under every log2 in the epic's math: log2(x) means log2(max(x, eps))
#define LS_EPS (1.0f / 1048576.0f)

typedef struct dt_iop_lightsurface_params_t
{
  float form;             // $MIN: -0.9 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "form"
  float relief;           // $MIN: -1.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "relief"
  float fill;             // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "fill"
  float light_node[9];    // $MIN: -2.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "light curve"
  float surface_contrast; // $MIN: -0.9 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "surface contrast"
  float surface_hue[8];   // $MIN: -2.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "surface lightness"
  // which decomposition model this edit was made with: dt_intrinsic_hash() of its id.
  // 0 = none recorded; reload_defaults() puts the active model's hash here
  uint32_t model_hash;    // $DEFAULT: 0
} dt_iop_lightsurface_params_t;

// the params are stored and hashed as raw bytes, so they must have no padding
_Static_assert(sizeof(dt_iop_lightsurface_params_t) == 22 * 4,
               "dt_iop_lightsurface_params_t must have no padding");
_Static_assert(sizeof(((dt_iop_lightsurface_params_t *)NULL)->light_node)
                 == LS_LIGHT_NODES * sizeof(float),
               "light_node[] must hold LS_LIGHT_NODES values");
_Static_assert(sizeof(((dt_iop_lightsurface_params_t *)NULL)->surface_hue)
                 == LS_HUE_NODES * sizeof(float),
               "surface_hue[] must hold LS_HUE_NODES values");

// the params, sanitized: same size, so the default params-sized piece->data fits
typedef dt_iop_lightsurface_params_t dt_iop_lightsurface_data_t;

#ifdef HAVE_AI
// which light layer a pipe needs. a piece hash holds op, instance, params and
// blending but no image id (src/develop/imageop.c:2446-2458), so two images
// whose upstream modules have equal params share an upstream hash; the image
// id keeps one image from taking the other's layer. the input size is not
// part of the key: it differs between the preview and the full pipe by
// design, and a crop change already changes the upstream hash
typedef struct _ls_key_t
{
  dt_imgid_t imgid;          // piece->pipe->image.id
  dt_hash_t upstream;        // _upstream_hash(): the modules before this one
  uint32_t model_hash;       // dt_intrinsic_hash() of the model that produced it
} _ls_key_t;

// the low-resolution light layer of epic §M1. immutable once created, which
// is what makes sharing it between pipe threads safe; it carries its model's
// constants, so the full-resolution code never looks the model up again
typedef struct _ls_layer_t
{
  gint refcount;             // g_atomic_int_*; the creator holds 1
  _ls_key_t key;             // DT_INVALID_HASH as upstream for local (non-shared) layers
  int w, h;                  // low-res size
  int src_w, src_h;          // size of the whole-image input it was computed from
  float K, rho_ref, y_floor; // epic §M1
  float *lambda, *lambda_c, *lambda_f;  // w*h each
  float *rho, *Y, *hue, *sat;           // w*h each; for §M4 and the GUI
  dt_intrinsic_model_params_t mp; // that model's per-model constants (copied)
  gboolean fell_back;        // the edit's model was missing; another one was used
  double ms;                 // inference wall time, for the status line and LS-14
  char *model_id;            // the id behind key.model_hash, for logs and dumps
} _ls_layer_t;

// one loaded model per id, shared by every instance and pipe. loading takes
// 13-44 s, so no model is loaded before a pipe needs one
typedef struct dt_iop_lightsurface_global_data_t
{
  dt_pthread_mutex_t lock;      // guards ctx_by_id; never held across a load
  dt_pthread_mutex_t load_lock; // serializes loads, so one model loads once
  dt_intrinsic_env_t *env;      // under load_lock
  GHashTable *ctx_by_id;        // model id -> dt_intrinsic_ctx_t*, or the failed sentinel
} dt_iop_lightsurface_global_data_t;
#endif

typedef struct dt_iop_lightsurface_gui_data_t
{
  GtkWidget *form, *relief, *fill;
  GtkWidget *light_node[LS_LIGHT_NODES];
  GtkWidget *surface_contrast;
  GtkWidget *surface_hue[LS_HUE_NODES];
#ifdef HAVE_AI
  // shared with the pipe threads, under the gui lock
  _ls_layer_t *layer;          // published by the preview pipe
  gboolean full_needs_refresh; // a canvas pipe rendered a placeholder
  _ls_key_t wanted;            // the key that placeholder was waiting for
#endif
} dt_iop_lightsurface_gui_data_t;

const char *name()
{
  return _("light and surface");
}

const char *aliases()
{
  return _("intrinsic|shading|albedo|relight|fill light|form|material");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self,
     _("edit the light falling on the scene and the surfaces it falls on separately,\n"
       "using an AI decomposition of the image"),
     _("creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_GRADING;
}

// no IOP_FLAGS_ALLOW_TILING: the light layer is computed from the whole image
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

// keeps the curve's output points x_i + v_i rising by at least
// LS_NODE_MIN_STEP per node, so tones never invert
static void _sanitize_nodes(float v[LS_LIGHT_NODES])
{
  for(int i = 1; i < LS_LIGHT_NODES; i++)
    v[i] = fmaxf(v[i], v[i - 1] - (1.0f - LS_NODE_MIN_STEP));
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_lightsurface_data_t *d = piece->data;
  memcpy(d, p1, sizeof(dt_iop_lightsurface_params_t));
  _sanitize_nodes(d->light_node);
}

void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_lightsurface_params_t *d = self->default_params;

  // a new edit records a concrete model rather than whatever is active later
  d->model_hash = 0;
#ifdef HAVE_AI
  char *id = dt_intrinsic_active_model_id();
  if(id) d->model_hash = dt_intrinsic_hash(id);
  g_free(id);
#endif
}

#ifdef HAVE_AI

/* --- model contexts --- */

// stands in the table for a model that failed to load, so it is not retried
static char _ls_failed_sentinel;
#define LS_CTX_FAILED ((gpointer)&_ls_failed_sentinel)

static void _ctx_free(gpointer data)
{
  if(data && data != LS_CTX_FAILED) dt_intrinsic_free(data);
}

void init_global(dt_iop_module_so_t *self)
{
  dt_iop_lightsurface_global_data_t *gd = calloc(1, sizeof(*gd));
  dt_pthread_mutex_init(&gd->lock, NULL);
  dt_pthread_mutex_init(&gd->load_lock, NULL);
  gd->ctx_by_id = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, _ctx_free);
  self->data = gd;
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_lightsurface_global_data_t *gd = self->data;
  if(!gd) return;
  g_hash_table_destroy(gd->ctx_by_id);
  dt_intrinsic_env_destroy(gd->env);
  dt_pthread_mutex_destroy(&gd->lock);
  dt_pthread_mutex_destroy(&gd->load_lock);
  free(gd);
  self->data = NULL;
}

// the model id to use for an edit (newly allocated), or NULL if none is
// installed; *fell_back is set when the edit asked for a model that is not
static char *_resolve_model_id(const uint32_t model_hash, gboolean *fell_back)
{
  *fell_back = FALSE;
  if(model_hash)
  {
    char *id = dt_intrinsic_model_id_for_hash(model_hash);
    if(id) return id;
  }

  *fell_back = (model_hash != 0);
  char *id = dt_intrinsic_active_model_id();
  if(id) return id;

  GList *models = dt_intrinsic_list_models();
  if(models) id = g_strdup(((dt_intrinsic_model_t *)models->data)->id);
  dt_intrinsic_models_free(models);
  return id;
}

// the loaded model for an id, loading it on first use; NULL if it cannot be
// loaded. blocks for the whole load, so only pipes that run inference call it
static dt_intrinsic_ctx_t *_get_ctx(dt_iop_module_t *self, const char *model_id)
{
  dt_iop_lightsurface_global_data_t *gd = self->global_data;

  dt_pthread_mutex_lock(&gd->load_lock);

  dt_pthread_mutex_lock(&gd->lock);
  gpointer value = NULL;
  const gboolean known = g_hash_table_lookup_extended(gd->ctx_by_id, model_id, NULL, &value);
  dt_pthread_mutex_unlock(&gd->lock);

  if(!known)
  {
    if(!gd->env) gd->env = dt_intrinsic_env_init();
    dt_intrinsic_ctx_t *ctx = gd->env ? dt_intrinsic_load(gd->env, model_id) : NULL;
    if(!ctx)
      dt_print(DT_DEBUG_ALWAYS, "[lightsurface] cannot load model %s", model_id);
    value = ctx ? (gpointer)ctx : LS_CTX_FAILED;
    dt_pthread_mutex_lock(&gd->lock);
    g_hash_table_insert(gd->ctx_by_id, g_strdup(model_id), value);
    dt_pthread_mutex_unlock(&gd->lock);
  }

  dt_pthread_mutex_unlock(&gd->load_lock);
  return value == LS_CTX_FAILED ? NULL : value;
}

// TRUE once a load of this model has failed. does not wait for a load in
// progress, so a canvas pipe can ask without stalling
static gboolean _ctx_failed(dt_iop_module_t *self, const char *model_id)
{
  dt_iop_lightsurface_global_data_t *gd = self->global_data;
  dt_pthread_mutex_lock(&gd->lock);
  const gboolean failed = g_hash_table_lookup(gd->ctx_by_id, model_id) == LS_CTX_FAILED;
  dt_pthread_mutex_unlock(&gd->lock);
  return failed;
}

/* --- the layer object --- */

static inline gboolean _key_equal(const _ls_key_t *a, const _ls_key_t *b)
{
  return a->imgid == b->imgid
    && a->upstream == b->upstream
    && a->model_hash == b->model_hash;
}

static _ls_layer_t *_layer_ref(_ls_layer_t *L)
{
  if(L) g_atomic_int_inc(&L->refcount);
  return L;
}

static void _layer_unref(_ls_layer_t *L)
{
  if(!L || !g_atomic_int_dec_and_test(&L->refcount)) return;
  dt_free_align(L->lambda);
  dt_free_align(L->lambda_c);
  dt_free_align(L->lambda_f);
  dt_free_align(L->rho);
  dt_free_align(L->Y);
  dt_free_align(L->hue);
  dt_free_align(L->sat);
  g_free(L->model_id);
  g_free(L);
}

/* --- debug dump --- */

static const char *_pipe_name(const dt_dev_pixelpipe_t *pipe)
{
  if(dt_pipe_is_image(pipe)) return "image";
  if(dt_pipe_is_export(pipe)) return "export";
  if(dt_pipe_is_thumb(pipe)) return "thumb";
  if(dt_pipe_is_preview(pipe)) return "preview";
  if(dt_pipe_is_preview2(pipe)) return "preview2";
  if(dt_pipe_is_full(pipe)) return "full";
  return "other";
}

// DT_LIGHTSURFACE_DUMP names a directory to write the layer and the
// full-resolution light into, as PFM, for comparison with the Python
// reference (LS-07, LS-08, LS-14); the prefix is <pipe>_<model>_<sequence>
static gchar *_dump_prefix(const dt_dev_pixelpipe_t *pipe, const char *model_id)
{
  const char *dir = g_getenv("DT_LIGHTSURFACE_DUMP");
  if(!dir || !dir[0]) return NULL;
  static gint seq = 0;
  return g_strdup_printf("%s/%s_%s_%04d", dir, _pipe_name(pipe), model_id,
                         g_atomic_int_add(&seq, 1));
}

// ch is 1 or 3 (4 writes the first three channels)
static void _dump(const gchar *prefix, const char *name, const float *buf,
                  const int w, const int h, const int ch)
{
  if(!prefix) return;
  gchar *fn = g_strdup_printf("%s_%s.pfm", prefix, name);
  dt_write_pfm(fn, w, h, buf, ch * sizeof(float));
  g_free(fn);
}

/* --- light layer, epic §M1 --- */

// nearest-rank percentile of the finite values (epic §M0); 0 if there are none
static int _float_cmp(const void *a, const void *b)
{
  const float x = *(const float *)a;
  const float y = *(const float *)b;
  return (x > y) - (x < y);
}

static float _percentile(const float *a, const size_t n, const double q)
{
  float *s = dt_alloc_align_float(n);
  if(!s) return 0.0f;
  size_t m = 0;
  for(size_t k = 0; k < n; k++)
    if(dt_isfinite(a[k])) s[m++] = a[k];
  float res = 0.0f;
  if(m)
  {
    qsort(s, m, sizeof(float), _float_cmp);
    res = s[(size_t)floor(q / 100.0 * (double)(m - 1))];
  }
  dt_free_align(s);
  return res;
}

// area-average downscale so the long edge is W_model, never upscaling
// (epic §M1.1). output pixel i averages the input pixels whose centers lie in
// [i/s, (i+1)/s); the same function as proto/ls_ref.py downscale_area()
static void _area_bins(const int n_out, const int n_in, const double s,
                       int *start, int *end)
{
  for(int i = 0; i < n_out; i++)
  {
    int a = MAX((int)ceil(i / s - 0.5), 0);
    int b = MIN((int)ceil((i + 1) / s - 0.5), n_in);
    if(b <= a) b = a + 1;
    start[i] = a;
    end[i] = b;
  }
}

static float *_downscale_area(const float *const in, const int W, const int H,
                              const int W_model, int *const w_out, int *const h_out)
{
  const double s = (double)W_model / MAX(W, H);
  if(s >= 1.0)
  {
    float *out = dt_alloc_align_float((size_t)W * H * 4);
    if(out) dt_iop_image_copy_by_size(out, in, W, H, 4);
    *w_out = W;
    *h_out = H;
    return out;
  }

  const int w = MAX(1, (int)floor(W * s + 0.5));
  const int h = MAX(1, (int)floor(H * s + 0.5));
  int *cs = g_new(int, w), *ce = g_new(int, w);
  int *rs = g_new(int, h), *re = g_new(int, h);
  _area_bins(w, W, s, cs, ce);
  _area_bins(h, H, s, rs, re);

  float *const out = dt_alloc_align_float((size_t)w * h * 4);
  if(out)
  {
    DT_OMP_FOR()
    for(int j = 0; j < h; j++)
      for(int i = 0; i < w; i++)
      {
        double acc[4] = { 0.0, 0.0, 0.0, 0.0 };
        for(int y = rs[j]; y < re[j]; y++)
          for(int x = cs[i]; x < ce[i]; x++)
            for(int c = 0; c < 4; c++)
              acc[c] += in[4 * ((size_t)y * W + x) + c];
        const double n = (double)(re[j] - rs[j]) * (ce[i] - cs[i]);
        for(int c = 0; c < 4; c++)
          out[4 * ((size_t)j * w + i) + c] = (float)(acc[c] / n);
      }
  }

  g_free(cs);
  g_free(ce);
  g_free(rs);
  g_free(re);
  *w_out = w;
  *h_out = h;
  return out;
}

// gray_guided(p, r, se) of epic §M0: the gray guided filter of p guided by
// itself. darktable's guided_filter() cannot do this, because it asserts a
// color guide (src/common/guided_filter.c:254)
static void _gray_guided(const float *const p, float *const out,
                         const int w, const int h, const int r, const float sqrt_eps)
{
  const size_t n = (size_t)w * h;
  float *const m = dt_alloc_align_float(n);
  float *const pp = dt_alloc_align_float(n);
  if(!m || !pp)
  {
    dt_free_align(m);
    dt_free_align(pp);
    memcpy(out, p, n * sizeof(float));
    return;
  }

  for(size_t k = 0; k < n; k++)
  {
    m[k] = p[k];
    pp[k] = p[k] * p[k];
  }
  // dt_box_mean() works in place
  dt_box_mean(m, h, w, 1, r, 1);
  dt_box_mean(pp, h, w, 1, r, 1);

  const float se2 = sqrt_eps * sqrt_eps;
  // reuse the buffers: m becomes a, pp becomes b
  for(size_t k = 0; k < n; k++)
  {
    const float v = fmaxf(0.0f, pp[k] - m[k] * m[k]);
    const float a = v / (v + se2);
    pp[k] = m[k] - a * m[k];
    m[k] = a;
  }
  dt_box_mean(m, h, w, 1, r, 1);
  dt_box_mean(pp, h, w, 1, r, 1);

  for(size_t k = 0; k < n; k++)
    out[k] = m[k] * p[k] + pp[k];

  dt_free_align(m);
  dt_free_align(pp);
}

// hue (degrees in [0, 360), the color equalizer's convention) and saturation
// of a working-RGB pixel, epic §M4.2. input_matrix is built as in
// src/iop/colorequal.c:1042-1044
static inline void _hue_sat(const dt_aligned_pixel_t rgb,
                            const dt_colormatrix_t input_matrix,
                            float *const h,
                            float *const s)
{
  dt_aligned_pixel_t XYZ_D65 = { 0.0f, 0.0f, 0.0f, 0.0f };
  dot_product(rgb, input_matrix, XYZ_D65);
  dt_aligned_pixel_t xyY = { 0.0f, 0.0f, 0.0f, 0.0f };
  dt_D65_XYZ_to_xyY(XYZ_D65, xyY);
  float UV[2];
  xyY_to_dt_UCS_UV(xyY, UV);

  // colorequal's ANGLE_SHIFT puts red at 0 (src/iop/colorequal.c:459)
  float hd = atan2f(UV[1], UV[0]) * (180.0f / (float)M_PI) - 20.0f;
  hd = fmodf(hd, 360.0f);
  if(hd < 0.0f) hd += 360.0f;
  if(hd >= 360.0f) hd -= 360.0f;
  *h = hd;

  // as colorequal.c:1066-1069, including its NORM_MIN thresholds
  const float dmin = fminf(rgb[0], fminf(rgb[1], rgb[2]));
  const float dmax = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2]));
  const float delta = dmax - dmin;
  *s = (dmax > NORM_MIN && delta > NORM_MIN) ? delta / dmax : 0.0f;
}

// epic §M1 on the whole-image input `in` (W_in x H_in, 4 channels, working
// RGB). runs the model, so it takes seconds to minutes; NULL on failure
static _ls_layer_t *_layer_compute(dt_iop_module_t *self,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   dt_intrinsic_ctx_t *ctx,
                                   const float *const in,
                                   const int W_in,
                                   const int H_in)
{
  const dt_iop_order_iccprofile_info_t *const wp
    = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  if(!wp) return NULL;

  const dt_intrinsic_model_params_t *const mp = dt_intrinsic_params(ctx);
  const char *model_id = dt_intrinsic_model_id(ctx);
  gchar *dump = _dump_prefix(piece->pipe, model_id);

  // 1. downscale (§M1.1)
  int w = 0, h = 0;
  float *const I_lr = _downscale_area(in, W_in, H_in, mp->working_size, &w, &h);
  if(!I_lr)
  {
    g_free(dump);
    return NULL;
  }
  const size_t n = (size_t)w * h;

  // 2. working RGB -> XYZ D50 -> linear sRGB, negatives clipped, HWC
  float *const rgb = dt_alloc_align_float(n * 3);
  float *const S = dt_alloc_align_float(n);
  _ls_layer_t *L = g_new0(_ls_layer_t, 1);
  L->lambda = dt_alloc_align_float(n);
  L->lambda_c = dt_alloc_align_float(n);
  L->lambda_f = dt_alloc_align_float(n);
  L->rho = dt_alloc_align_float(n);
  L->Y = dt_alloc_align_float(n);
  L->hue = dt_alloc_align_float(n);
  L->sat = dt_alloc_align_float(n);
  L->refcount = 1;
  if(!rgb || !S || !L->lambda || !L->lambda_c || !L->lambda_f
     || !L->rho || !L->Y || !L->hue || !L->sat)
    goto fail;

  DT_OMP_FOR()
  for(size_t k = 0; k < n; k++)
  {
    dt_aligned_pixel_t lin = { 0.0f, 0.0f, 0.0f, 0.0f };
    if(wp->nonlinearlut)
      dt_ioppr_apply_trc(I_lr + 4 * k, lin, wp->lut_in, wp->unbounded_coeffs_in, wp->lutsize);
    else
      for(int c = 0; c < 3; c++) lin[c] = I_lr[4 * k + c];
    dt_aligned_pixel_t XYZ = { 0.0f, 0.0f, 0.0f, 0.0f };
    dot_product(lin, wp->matrix_in, XYZ);
    dt_aligned_pixel_t srgb = { 0.0f, 0.0f, 0.0f, 0.0f };
    dt_XYZ_to_Rec709_D50(XYZ, srgb);
    for(int c = 0; c < 3; c++)
      rgb[3 * k + c] = dt_isfinite(srgb[c]) ? fmaxf(srgb[c], 0.0f) : 0.0f;
  }

  // 3. the model (§M1.2)
  const double t0 = dt_get_wtime();
  if(!dt_intrinsic_shading(ctx, rgb, w, h, S))
  {
    dt_print(DT_DEBUG_ALWAYS, "[lightsurface] %s: inference with %s failed",
             _pipe_name(piece->pipe), model_id);
    goto fail;
  }
  L->ms = 1000.0 * (dt_get_wtime() - t0);

  // 4. key light and surface (§M1.3-4)
  for(size_t k = 0; k < n; k++)
    L->lambda[k] = log2f(fmaxf(S[k], LS_EPS));
  L->K = _percentile(L->lambda, n, LS_KEY_PERCENTILE);
  for(size_t k = 0; k < n; k++)
  {
    L->lambda[k] -= L->K;
    const float Y = dt_ioppr_get_rgb_matrix_luminance(I_lr + 4 * k, wp->matrix_in, wp->lut_in,
                                                      wp->unbounded_coeffs_in, wp->lutsize,
                                                      wp->nonlinearlut);
    L->Y[k] = Y;
    L->rho[k] = log2f(fmaxf(Y, LS_EPS)) - L->lambda[k];
  }
  L->rho_ref = _percentile(L->rho, n, 50.0);

  // 5. form/relief split, edge-aware (§M1.5)
  const int r_s = MAX(1, (int)roundf(mp->split_radius_rel * MAX(w, h)));
  _gray_guided(L->lambda, L->lambda_c, w, h, r_s, mp->split_sqrt_eps);
  for(size_t k = 0; k < n; k++)
    L->lambda_f[k] = L->lambda[k] - L->lambda_c[k];

  // 6. hue and saturation (§M4.2)
  dt_colormatrix_t input_matrix;
  dt_colormatrix_mul(input_matrix, XYZ_D50_to_D65_CAT16, wp->matrix_in);
  DT_OMP_FOR()
  for(size_t k = 0; k < n; k++)
    _hue_sat(I_lr + 4 * k, input_matrix, L->hue + k, L->sat + k);

  // 7. the log guide's floor (§M1.7)
  L->y_floor = exp2f(-LS_GUIDE_FLOOR_EV) * fmaxf(_percentile(L->Y, n, 50.0), LS_EPS);

  L->w = w;
  L->h = h;
  L->src_w = W_in;
  L->src_h = H_in;
  L->mp = *mp;
  L->model_id = g_strdup(model_id);

  dt_print(DT_DEBUG_AI | DT_DEBUG_PIPE,
           "[lightsurface] %s: light layer %dx%d from %dx%d with %s in %.0f ms,"
           " K %.3f, rho_ref %.3f",
           _pipe_name(piece->pipe), w, h, W_in, H_in, model_id, L->ms, L->K, L->rho_ref);

  if(dump)
  {
    _dump(dump, "I_lr", rgb, w, h, 3);
    _dump(dump, "I_lr_work", I_lr, w, h, 4);
    _dump(dump, "S", S, w, h, 1);
    _dump(dump, "lambda", L->lambda, w, h, 1);
    _dump(dump, "lambda_c", L->lambda_c, w, h, 1);
    _dump(dump, "lambda_f", L->lambda_f, w, h, 1);
    gchar *fn = g_strdup_printf("%s_layer.txt", dump);
    FILE *f = g_fopen(fn, "w");
    if(f)
    {
      fprintf(f, "w %d\nh %d\nsrc_w %d\nsrc_h %d\nK %.9g\nrho_ref %.9g\ny_floor %.9g\n"
              "ms %.1f\nworking_size %d\nsplit_radius_rel %.9g\nsplit_sqrt_eps %.9g\n"
              "guide_radius_factor %.9g\nguide_sqrt_eps %.9g\nr_s %d\n",
              w, h, W_in, H_in, L->K, L->rho_ref, L->y_floor, L->ms, mp->working_size,
              mp->split_radius_rel, mp->split_sqrt_eps, mp->guide_radius_factor,
              mp->guide_sqrt_eps, r_s);
      fclose(f);
    }
    g_free(fn);
  }

  g_free(dump);
  dt_free_align(I_lr);
  dt_free_align(rgb);
  dt_free_align(S);
  return L;

fail:
  g_free(dump);
  dt_free_align(I_lr);
  dt_free_align(rgb);
  dt_free_align(S);
  _layer_unref(L);
  return NULL;
}

/* --- full resolution, epic §M2 --- */

// G(I) of epic §M0: 4 channels, log2(max(I_c, y_floor)), G_3 = 0
static void _log_guide(const float *const in, const size_t npix,
                       const float y_floor, float *const guide)
{
  DT_OMP_FOR()
  for(size_t k = 0; k < npix; k++)
  {
    for(int c = 0; c < 3; c++)
      guide[4 * k + c] = log2f(fmaxf(in[4 * k + c], y_floor));
    guide[4 * k + 3] = 0.0f;
  }
}

// bilinear samples of a low-res field (lw x lh) at the ROI's pixel centers
// (epic §M2.1-2); buf_in is the module's full input at scale 1
static void _sample_lr(const float *const lr, const int lw, const int lh,
                       const dt_iop_roi_t *const roi_in,
                       const dt_dev_pixelpipe_iop_t *const piece,
                       float *const out)
{
  const int width = roi_in->width;
  const int height = roi_in->height;
  const float sx = roi_in->scale * piece->buf_in.width;
  const float sy = roi_in->scale * piece->buf_in.height;
  const float ox = roi_in->x;
  const float oy = roi_in->y;

  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
  {
    const float v = (oy + y + 0.5f) / sy;
    const float Yc = CLAMP(v * lh - 0.5f, 0.0f, (float)(lh - 1));
    const int y0 = (int)floorf(Yc);
    const int y1 = MIN(y0 + 1, lh - 1);
    const float fy = Yc - y0;
    for(int x = 0; x < width; x++)
    {
      const float u = (ox + x + 0.5f) / sx;
      const float Xc = CLAMP(u * lw - 0.5f, 0.0f, (float)(lw - 1));
      const int x0 = (int)floorf(Xc);
      const int x1 = MIN(x0 + 1, lw - 1);
      const float fx = Xc - x0;
      const float top = lr[(size_t)y0 * lw + x0] * (1.0f - fx) + lr[(size_t)y0 * lw + x1] * fx;
      const float bot = lr[(size_t)y1 * lw + x0] * (1.0f - fx) + lr[(size_t)y1 * lw + x1] * fx;
      out[(size_t)y * width + x] = top * (1.0f - fy) + bot * fy;
    }
  }
}

// a low-res field of layer L brought to the ROI (epic §M2.3, and §M2.5 for
// dS_lr in LS-08): bilinear samples, then, where the ROI is finer than the
// layer, the guided filter with the log guide snaps them to image edges
static void _sample_refine(const _ls_layer_t *const L, const float *const lr,
                           const float *const guide,
                           const dt_iop_roi_t *const roi_in,
                           const dt_dev_pixelpipe_iop_t *const piece,
                           float *const out)
{
  const int width = roi_in->width;
  const int height = roi_in->height;
  const float f = roi_in->scale * piece->buf_in.width / L->w;

  if(f <= 1.0f)
  {
    _sample_lr(lr, L->w, L->h, roi_in, piece, out);
    return;
  }

  float *const up = dt_alloc_align_float((size_t)width * height);
  if(!up)
  {
    _sample_lr(lr, L->w, L->h, roi_in, piece, out);
    return;
  }
  _sample_lr(lr, L->w, L->h, roi_in, piece, up);
  const int r = MAX(1, (int)ceilf(L->mp.guide_radius_factor * f));
  guided_filter(guide, up, out, width, height, 4, r, L->mp.guide_sqrt_eps,
                1.0f, -FLT_MAX, FLT_MAX);
  dt_free_align(up);
}

// lambda (refined) and lambda_f (sampled only: it holds low-amplitude shading
// by construction, so its misregistration is small) for the ROI, epic
// §M2.1-4; lambda_c = lambda - lambda_f is formed where it is used
static void _light_full(const _ls_layer_t *const L, const float *const guide,
                        const dt_iop_roi_t *const roi_in,
                        const dt_dev_pixelpipe_iop_t *const piece,
                        float *const lambda, float *const lambda_f)
{
  _sample_refine(L, L->lambda, guide, roi_in, piece, lambda);
  _sample_lr(L->lambda_f, L->w, L->h, roi_in, piece, lambda_f);
}

/* --- which pipe does what (LS-07 Part C) --- */

// the ROI covers the whole image: this pipe can compute a layer itself
static gboolean _roi_is_whole(const dt_iop_roi_t *const roi_in,
                              const dt_dev_pixelpipe_iop_t *const piece)
{
  return roi_in->x == 0 && roi_in->y == 0
    && fabsf(roi_in->width - roi_in->scale * piece->buf_in.width) <= 2.0f
    && fabsf(roi_in->height - roi_in->scale * piece->buf_in.height) <= 2.0f;
}

// the upstream part of the layer key: dt_dev_hash_plus(..., BACK_EXCL) without
// the hidden modules. those are enabled per pipe, not per history:
// demosaicscale only resamples, on canvas pipes of raw files
// (src/iop/demosaicscale.c:151-160), and rawoverexposed is the full pipe's
// clipping indicator (src/iop/rawoverexposed.c:392). counting them gives the
// preview and the full pipe different hashes for the same history, and the
// full pipe would never find the preview's layer. rotatepixels (fixed per
// camera, and the image id is in the key) and mask_manager are skipped too
static dt_hash_t _upstream_hash(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe)
{
  dt_develop_t *dev = self->dev;
  dt_hash_t hash = DT_INITHASH;
  dt_pthread_mutex_lock(&dev->history_mutex);
  GList *modules = g_list_last(pipe->iop);
  GList *pieces = g_list_last(pipe->nodes);
  while(modules && pieces)
  {
    const dt_iop_module_t *module = modules->data;
    const dt_dev_pixelpipe_iop_t *piece = pieces->data;
    if(piece->enabled
       && module->iop_order < self->iop_order
       && !(module->flags() & IOP_FLAGS_HIDDEN))
      hash = dt_hash(hash, &piece->hash, sizeof(piece->hash));
    modules = g_list_previous(modules);
    pieces = g_list_previous(pieces);
  }
  dt_pthread_mutex_unlock(&dev->history_mutex);
  return hash;
}

// a ref of the published layer if its key is `key`, else NULL
static _ls_layer_t *_take_published(dt_iop_module_t *self, const _ls_key_t *key)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  dt_iop_gui_enter_critical_section(self);
  _ls_layer_t *L = g->layer && _key_equal(&g->layer->key, key) ? _layer_ref(g->layer) : NULL;
  dt_iop_gui_leave_critical_section(self);
  return L;
}

// the light layer this process() call uses (a ref the caller releases), or
// NULL when the output stays an identity
static _ls_layer_t *_layer_for_pipe(dt_iop_module_t *self,
                                    dt_dev_pixelpipe_iop_t *piece,
                                    const float *const in,
                                    const dt_iop_roi_t *const roi_in)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  const gboolean gui = self->dev->gui_attached && g;
  const dt_iop_lightsurface_data_t *const d = piece->data;
  dt_dev_pixelpipe_t *const pipe = piece->pipe;
  const char *pname = _pipe_name(pipe);
  const gboolean whole = _roi_is_whole(roi_in, piece);

  dt_print(DT_DEBUG_PIPE,
           "[lightsurface] %s: roi_in %d,%d %dx%d scale %.5f, buf_in %dx%d, whole image %s",
           pname, roi_in->x, roi_in->y, roi_in->width, roi_in->height, roi_in->scale,
           piece->buf_in.width, piece->buf_in.height, whole ? "yes" : "no");

  gboolean fell_back = FALSE;
  char *model_id = _resolve_model_id(d->model_hash, &fell_back);
  if(!model_id)
  {
    if(!gui)
      dt_print(DT_DEBUG_ALWAYS,
               "[lightsurface] no intrinsic decomposition model is available"
               " (AI disabled or none installed), output unchanged");
    return NULL;
  }
  if(fell_back && !gui)
    dt_print(DT_DEBUG_ALWAYS,
             "[lightsurface] model of this edit is not installed, using %s", model_id);

  // the model hash of a ctx is dt_intrinsic_hash() of its id, so a pipe can
  // form its key without loading the model
  _ls_key_t key = { .imgid = pipe->image.id,
                    .upstream = DT_INVALID_HASH,
                    .model_hash = dt_intrinsic_hash(model_id) };
  _ls_layer_t *L = NULL;

  if(gui && (dt_pipe_is_preview(pipe) || dt_pipe_is_canvas(pipe)))
  {
    key.upstream = _upstream_hash(self, pipe);
    dt_print(DT_DEBUG_PIPE,
             "[lightsurface] %s: key image %d, upstream %016" PRIx64 ", model %08x (%s)",
             pname, key.imgid, key.upstream, key.model_hash, model_id);
  }

  if(gui && dt_pipe_is_preview(pipe))
  {
    // the preview sees the whole image: it computes the layer and publishes it
    L = _take_published(self, &key);
    if(!L)
    {
      dt_intrinsic_ctx_t *ctx = _get_ctx(self, model_id);
      if(ctx) L = _layer_compute(self, piece, ctx, in, roi_in->width, roi_in->height);
      if(L)
      {
        L->key = key;
        L->fell_back = fell_back;
        dt_iop_gui_enter_critical_section(self);
        _ls_layer_t *old = g->layer;
        g->layer = _layer_ref(L);
        dt_iop_gui_leave_critical_section(self);
        _layer_unref(old);
      }
    }
  }
  else if(gui && dt_pipe_is_canvas(pipe))
  {
    // the full pipe and the second window's never run inference, not even in
    // late-scaling mode where they see the whole image (unlike
    // hazeremoval.c:598): that would be a second inference per change, queued
    // behind the preview's on the same model. they take the preview's layer
    // when its key matches their own, and render a placeholder until it does.
    // there is no dt_dev_sync_pixelpipe_hash() wait: it compares against
    // dt_dev_hash_plus(), not this key, and inference takes far longer than
    // its timeout anyway
    L = _take_published(self, &key);
    if(!L && !_ctx_failed(self, model_id))
    {
      // placeholder: the identity output is cached under this module's
      // params hash, which will not change when the layer arrives, so the
      // cache lines from here on are dropped and the preview-finished
      // handler reprocesses the center once the layer is there
      dt_dev_pixelpipe_cache_invalidate_later(pipe, self->iop_order,
                                              "lightsurface: waiting for the light layer ");
      dt_iop_gui_enter_critical_section(self);
      g->full_needs_refresh = TRUE;
      g->wanted = key;
      dt_iop_gui_leave_critical_section(self);
      dt_print(DT_DEBUG_PIPE, "[lightsurface] %s: no light layer yet, placeholder", pname);
    }
  }
  else if(!whole)
  {
    // export, thumbnails and cli always see the whole image; this is a
    // dt_dev_image() pipe (a snapshot or the slideshow) rendering a zoomed view
    dt_print(DT_DEBUG_ALWAYS,
             "[lightsurface] %s pipe sees only part of the image, so there is no"
             " light layer; output unchanged", pname);
  }
  else
  {
    // export, thumbnails, darktable-cli and dt_dev_image() pipes: no gui
    // data to share through, so compute a local layer
    dt_intrinsic_ctx_t *ctx = _get_ctx(self, model_id);
    if(ctx) L = _layer_compute(self, piece, ctx, in, roi_in->width, roi_in->height);
    if(L)
    {
      L->key = key;
      L->fell_back = fell_back;
    }
    else
      dt_print(DT_DEBUG_ALWAYS,
               "[lightsurface] no light layer with %s, output unchanged", model_id);
  }

  if(L)
    dt_print(DT_DEBUG_PIPE, "[lightsurface] %s: using the light layer %dx%d of %s",
             pname, L->w, L->h, model_id);

  g_free(model_id);
  return L;
}

// the preview published a layer: redraw the canvases that rendered a
// placeholder for it. runs in the GUI thread
static void _preview_pipe_finished_callback(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  if(!g) return;

  // only when the layer is the one the placeholder waited for: otherwise a
  // canvas whose key never matches would reprocess after every preview run,
  // and each reprocess reruns the preview
  dt_iop_gui_enter_critical_section(self);
  const gboolean refresh = g->full_needs_refresh && g->layer
    && _key_equal(&g->layer->key, &g->wanted);
  if(refresh) g->full_needs_refresh = FALSE;
  dt_iop_gui_leave_critical_section(self);
  if(!refresh) return;

  dt_develop_t *dev = self->dev;
  dt_print(DT_DEBUG_PIPE, "[lightsurface] light layer arrived, reprocessing the center");
  // dt_dev_reprocess_center() obsoletes only the full pipe's cache; the second
  // window's placeholder lines go the same way
  if(dev->preview2.pipe)
    dev->preview2.pipe->cache_obsolete_order = self->iop_order;
  dt_dev_reprocess_center(dev, self->iop_order);
}

#endif // HAVE_AI

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  if(!dt_iop_have_required_input_format(4, self, piece->colors,
                                        ivoid, ovoid, roi_in, roi_out))
    return;

  const float *const in = (const float *)ivoid;
  float *const out = (float *)ovoid;

  // the output is an identity until the light and surface math lands (LS-08)
  dt_iop_image_copy_by_size(out, in, roi_out->width, roi_out->height, 4);

#ifdef HAVE_AI
  // the module does not change the ROI, so in, out and the §M2 buffers share it
  assert(roi_in->width == roi_out->width && roi_in->height == roi_out->height);

  _ls_layer_t *L = _layer_for_pipe(self, piece, in, roi_in);
  if(!L) return;

  const size_t npix = (size_t)roi_in->width * roi_in->height;
  float *const guide = dt_alloc_align_float(npix * 4);
  float *const lambda = dt_alloc_align_float(npix);
  float *const lambda_f = dt_alloc_align_float(npix);
  if(guide && lambda && lambda_f)
  {
    _log_guide(in, npix, L->y_floor, guide);
    _light_full(L, guide, roi_in, piece, lambda, lambda_f);

    gchar *dump = _dump_prefix(piece->pipe, L->model_id);
    if(dump)
    {
      _dump(dump, "lambda_full", lambda, roi_in->width, roi_in->height, 1);
      gchar *fn = g_strdup_printf("%s_full.txt", dump);
      FILE *f = g_fopen(fn, "w");
      if(f)
      {
        fprintf(f, "roi_x %d\nroi_y %d\nroi_w %d\nroi_h %d\nroi_scale %.9g\n"
                "buf_in_w %d\nbuf_in_h %d\nlayer_w %d\nlayer_h %d\ny_floor %.9g\n",
                roi_in->x, roi_in->y, roi_in->width, roi_in->height, roi_in->scale,
                piece->buf_in.width, piece->buf_in.height, L->w, L->h, L->y_floor);
        fclose(f);
      }
      g_free(fn);
      g_free(dump);
    }
  }

  dt_free_align(guide);
  dt_free_align(lambda);
  dt_free_align(lambda_f);
  _layer_unref(L);
#endif
}

void gui_init(dt_iop_module_t *self)
{
  static const char *const node_labels[LS_LIGHT_NODES] =
    { N_("-7 EV"), N_("-6 EV"), N_("-5 EV"), N_("-4 EV"), N_("-3 EV"),
      N_("-2 EV"), N_("-1 EV"), N_("+0 EV"), N_("+1 EV") };
  static const char *const hue_labels[LS_HUE_NODES] =
    { N_("red"), N_("orange"), N_("yellow"), N_("green"),
      N_("cyan"), N_("blue"), N_("lavender"), N_("magenta") };

  dt_iop_lightsurface_gui_data_t *g = IOP_GUI_ALLOC(lightsurface);

  self->widget = dt_gui_vbox();

  g->form = dt_bauhaus_slider_from_params(self, "form");
  g->relief = dt_bauhaus_slider_from_params(self, "relief");
  g->fill = dt_bauhaus_slider_from_params(self, "fill");

  for(int i = 0; i < LS_LIGHT_NODES; i++)
  {
    gchar *field = g_strdup_printf("light_node[%d]", i);
    g->light_node[i] = dt_bauhaus_slider_from_params(self, field);
    g_free(field);
    dt_bauhaus_widget_set_label(g->light_node[i], N_("light curve"), node_labels[i]);
    dt_bauhaus_slider_set_format(g->light_node[i], _(" EV"));
  }

  g->surface_contrast = dt_bauhaus_slider_from_params(self, "surface_contrast");

  for(int j = 0; j < LS_HUE_NODES; j++)
  {
    gchar *field = g_strdup_printf("surface_hue[%d]", j);
    g->surface_hue[j] = dt_bauhaus_slider_from_params(self, field);
    g_free(field);
    dt_bauhaus_widget_set_label(g->surface_hue[j], N_("surface lightness"), hue_labels[j]);
    dt_bauhaus_slider_set_format(g->surface_hue[j], _(" EV"));
  }

#ifdef HAVE_AI
  g->layer = NULL;
  g->full_needs_refresh = FALSE;
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED,
                           _preview_pipe_finished_callback);
#endif
}

// the signal is disconnected by dt_iop_gui_cleanup_module() before this runs
// (DT_CONTROL_SIGNAL_DISCONNECT_ALL, src/develop/imageop.c:2470)
void gui_cleanup(dt_iop_module_t *self)
{
#ifdef HAVE_AI
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  dt_iop_gui_enter_critical_section(self);
  _ls_layer_t *L = g->layer;
  g->layer = NULL;
  dt_iop_gui_leave_critical_section(self);
  _layer_unref(L);
#endif
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
