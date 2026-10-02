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
#include "common/colorspaces.h"
#include "common/imagebuf.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/paint.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#ifdef HAVE_AI
#include "common/ai/intrinsic.h"
#include "common/ai_models.h"
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
// the total gain is clamped to this many EV, both directions (epic §M5)
#define LS_MAX_GAIN_EV 8.0f
// the log guide's floor, in EV below the median luminance (epic §M1.7)
#define LS_GUIDE_FLOOR_EV 10.0f
// percentile of log shading that defines the key light (epic §M1.3)
#define LS_KEY_PERCENTILE 90.0
// the floor under every log2 in the epic's math: log2(x) means log2(max(x, eps))
#define LS_EPS (1.0f / 1048576.0f)

// the light histograms of epic §M6: 0.1 EV bins over [-12, +2] EV
#define LS_HIST_BINS 140
#define LS_HIST_MIN (-12.0f)
#define LS_HIST_MAX 2.0f
// the light graph shows light levels in [-8, +2] EV on x, and corrections in
// [-2, +2] EV on y, the range of the node params
#define LS_GRAPH_X_MIN (-8.0f)
#define LS_GRAPH_X_MAX 2.0f
#define LS_GRAPH_Y_MAX 2.0f

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
  float cov_x, cov_y;        // fraction of the image that input spans, per axis:
                             // below 1 for a high-quality export (_roi_is_whole())
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

// what the center view shows; epic §M7
typedef enum dt_iop_lightsurface_display_t
{
  LS_DISPLAY_NORMAL = 0,
  LS_DISPLAY_LIGHT,
  LS_DISPLAY_SURFACE,
  LS_DISPLAY_CHANGES
} dt_iop_lightsurface_display_t;

typedef struct dt_iop_lightsurface_gui_data_t
{
  GtkNotebook *notebook;
  GtkDrawingArea *area;
  GtkWidget *ratio_label;
  GtkWidget *form, *relief, *fill;
  dt_gui_collapsible_section_t nodes_section;
  GtkWidget *light_node[LS_LIGHT_NODES];
  GtkWidget *surface_contrast;
  GtkWidget *surface_hue[LS_HUE_NODES];
  GtkWidget *model, *display, *warn_toggle, *status_label, *map_box;
  uint32_t *model_hashes;  // index-aligned with the first n_models entries of `model`
  int n_models;            // an entry past these is the "not installed" one, not a choice
  // what is shown, not what is edited: never in the params, so never in the
  // history. under the gui lock, since the full pipe reads them
  dt_iop_lightsurface_display_t display_mode;
  gboolean show_warn;
  // the light graph, GUI thread only. the histograms count light levels
  // (epic §M6) before and after the light edit
  float hist_before[LS_HIST_BINS], hist_after[LS_HIST_BINS];
  gboolean hist_valid;     // a layer of the current image and model was found
  float ratio_before, ratio_after;
  int hovered_node, dragged_node; // -1 for none
  float drag_offset;       // node value minus the pointer's, in EV, at the press
  gboolean pointer_in;     // the pointer is over the graph, at pointer_x
  double pointer_x;        // in graph coordinates, for scrolling
  // the light level under the pointer on the image, drawn on the graph
  float cursor_lambda;
  gboolean cursor_lambda_valid;
#ifdef HAVE_AI
  // shared with the pipe threads, under the gui lock
  _ls_layer_t *layer;          // published by the preview pipe
  gboolean full_needs_refresh; // a canvas pipe rendered a placeholder
  _ls_key_t wanted;            // the key that placeholder was waiting for
  gboolean analyzing;          // the preview is computing a layer
#endif
} dt_iop_lightsurface_gui_data_t;

static void _update_status(dt_iop_module_t *self);
static void _update_stats(dt_iop_module_t *self);

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
  dt_pthread_mutex_init(&gd->lock);
  dt_pthread_mutex_init(&gd->load_lock);
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

// epic §M1 on the whole-image input `in` (the ROI's size, 4 channels, working
// RGB). runs the model, so it takes seconds to minutes; NULL on failure
static _ls_layer_t *_layer_compute(dt_iop_module_t *self,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   dt_intrinsic_ctx_t *ctx,
                                   const float *const in,
                                   const dt_iop_roi_t *const roi_in)
{
  const int W_in = roi_in->width;
  const int H_in = roi_in->height;
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
  L->cov_x = W_in / (roi_in->scale * piece->buf_in.width);
  L->cov_y = H_in / (roi_in->scale * piece->buf_in.height);
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
      fprintf(f, "w %d\nh %d\nsrc_w %d\nsrc_h %d\ncov_x %.9g\ncov_y %.9g\n"
              "K %.9g\nrho_ref %.9g\ny_floor %.9g\n"
              "ms %.1f\nworking_size %d\nsplit_radius_rel %.9g\nsplit_sqrt_eps %.9g\n"
              "guide_radius_factor %.9g\nguide_sqrt_eps %.9g\nr_s %d\n",
              w, h, W_in, H_in, L->cov_x, L->cov_y,
              L->K, L->rho_ref, L->y_floor, L->ms, mp->working_size,
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

// bilinear samples of a low-res field of layer L (L->w x L->h) at the ROI's
// pixel centers (epic §M2.1-2); buf_in is the module's full input at scale 1.
// the field spans only L's coverage of the image, so that is what maps to it
static void _sample_lr(const _ls_layer_t *const L, const float *const lr,
                       const dt_iop_roi_t *const roi_in,
                       const dt_dev_pixelpipe_iop_t *const piece,
                       float *const out)
{
  const int width = roi_in->width;
  const int height = roi_in->height;
  const int lw = L->w;
  const int lh = L->h;
  const float sx = roi_in->scale * piece->buf_in.width * L->cov_x;
  const float sy = roi_in->scale * piece->buf_in.height * L->cov_y;
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
  const float f = roi_in->scale * piece->buf_in.width * L->cov_x / L->w;

  if(f <= 1.0f)
  {
    _sample_lr(L, lr, roi_in, piece, out);
    return;
  }

  float *const up = dt_alloc_align_float((size_t)width * height);
  if(!up)
  {
    _sample_lr(L, lr, roi_in, piece, out);
    return;
  }
  _sample_lr(L, lr, roi_in, piece, up);
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
  _sample_lr(L, L->lambda_f, roi_in, piece, lambda_f);
}

/* --- the edits, epic §M3-§M5 --- */

// every edit below is a delta built from terms that are each an exact zero at
// default params (a product with a zero param, or a curve correction with zero
// coefficients). a "new minus old" difference would not be: in float32,
// lc + (lambda - lc) != lambda for about 2 % of inputs, so the module would
// stop being an identity at defaults (epic §5)

#endif // HAVE_AI

// the light curve and the light delta are needed without AI too: the light
// graph draws them

// secants and tangents of the node curve F, epic §M3.3. v must be sanitized
// (_sanitize_nodes(), which commit_params() applies): only then are all the
// secants at least LS_NODE_MIN_STEP and the curve defined
static void _curve_tangents(const float v[LS_LIGHT_NODES],
                            float m[LS_LIGHT_NODES])
{
  float sec[LS_LIGHT_NODES - 1];
  for(int k = 0; k < LS_LIGHT_NODES - 1; k++)
    sec[k] = 1.0f + v[k + 1] - v[k];

  // the ends: slope 1, to meet the straight continuation, unless the end
  // segment is too steep for that to stay monotone
  m[0] = fminf(1.0f, 2.0f * sec[0]);
  for(int k = 1; k < LS_LIGHT_NODES - 1; k++)
    m[k] = 2.0f / (1.0f / sec[k - 1] + 1.0f / sec[k]);
  m[LS_LIGHT_NODES - 1] = fminf(1.0f, 2.0f * sec[LS_LIGHT_NODES - 2]);
}

// C(x) = F(x) - x of epic §M3.3, computed directly rather than as F(x) - x:
// the Hermite basis reproduces straight lines, so the line's share cancels
// in the coefficients. nodes at x_i = -7 + i, m from _curve_tangents()
static inline float _light_correction(const float x,
                                      const float v[LS_LIGHT_NODES],
                                      const float m[LS_LIGHT_NODES])
{
  // the segment, clamped to the node range; fmaxf() also sends NaN to 0
  const float kf = fminf(fmaxf(floorf(x + 7.0f), 0.0f), (float)(LS_LIGHT_NODES - 2));
  const int k = (int)kf;
  const float t = x - (kf - 7.0f);
  const float t2 = t * t;
  const float t3 = t2 * t;
  const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
  const float h10 = t3 - 2.0f * t2 + t;
  const float h01 = -2.0f * t3 + 3.0f * t2;
  const float h11 = t3 - t2;
  const float C = h00 * v[k] + h10 * (m[k] - 1.0f) + h01 * v[k + 1] + h11 * (m[k + 1] - 1.0f);

  // outside the nodes F continues with slope 1
  return x < -7.0f ? v[0] : (x > 1.0f ? v[LS_LIGHT_NODES - 1] : C);
}

// dL = a_1 + a_2 + a_3 of epic §M3. d is the committed data, so its nodes are
// already sanitized, and m holds their tangents
static inline float _light_delta(const float lambda,
                                 const float lambda_c,
                                 const float lambda_f,
                                 const dt_iop_lightsurface_data_t *const d,
                                 const float m[LS_LIGHT_NODES])
{
  // form and relief
  const float a1 = d->form * lambda_c + d->relief * lambda_f;
  const float lambda1 = lambda + a1;
  // the node curve
  const float a2 = _light_correction(lambda1, d->light_node, m);
  const float lambda2 = lambda1 + a2;
  // fill: log2((2^lambda2 + phi) / (1 + phi)) - lambda2, written so that it is
  // exactly 0 at phi = 0. the fminf() only keeps exp2f() finite
  const float phi = d->fill;
  const float a3 = log2f(1.0f + phi * exp2f(fminf(-lambda2, 100.0f))) - log2f(1.0f + phi);
  return a1 + a2 + a3;
}

#ifdef HAVE_AI

// H(h) of epic §M4.3: the periodic Catmull-Rom spline through (45 j, q_j)
static inline float _hue_curve(const float h_deg, const float q[LS_HUE_NODES])
{
  float hw = fmodf(h_deg, 360.0f);
  if(hw < 0.0f) hw += 360.0f;
  // a tiny negative h wraps to exactly 360
  if(hw >= 360.0f) hw -= 360.0f;
  // fmaxf() sends NaN to segment 0
  const float jf = fminf(fmaxf(floorf(hw / 45.0f), 0.0f), (float)(LS_HUE_NODES - 1));
  const int j = (int)jf;
  const float t = (hw - 45.0f * jf) / 45.0f;

  const float p0 = q[(j + LS_HUE_NODES - 1) % LS_HUE_NODES];
  const float p1 = q[j];
  const float p2 = q[(j + 1) % LS_HUE_NODES];
  const float p3 = q[(j + 2) % LS_HUE_NODES];
  const float t2 = t * t;
  const float t3 = t2 * t;
  return 0.5f * (2.0f * p1 + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2
                 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

// dS_lr of epic §M4.4 for one low-res pixel: sc (rho - rho_ref) is a product
// with the param, not an edited rho minus the old one
static inline float _surface_delta(const float Y,
                                   const float rho,
                                   const float rho_ref,
                                   const float sc,
                                   const float h_deg,
                                   const float s,
                                   const float q[LS_HUE_NODES])
{
  if(Y <= LS_EPS) return 0.0f;
  const float w_c = fminf(1.0f, s / LS_SAT_FULL);
  return sc * (rho - rho_ref) + w_c * _hue_curve(h_deg, q);
}

// the total gain in EV, epic §M5: clamped first, so an infinite delta keeps
// its sign at the clamp, and only a NaN is rejected
static inline float _gain(const float dL, const float dS)
{
  const float x = dL + dS;
  // comparisons rather than fminf()/fmaxf(), which would turn a NaN into a bound
  const float g = x > LS_MAX_GAIN_EV ? LS_MAX_GAIN_EV : (x < -LS_MAX_GAIN_EV ? -LS_MAX_GAIN_EV : x);
  return dt_isfinite(g) ? g : 0.0f;
}

// all 21 float params zero (-0.0f included): the edits are an identity
static gboolean _params_are_zero(const dt_iop_lightsurface_data_t *const d)
{
  if(d->form != 0.0f || d->relief != 0.0f || d->fill != 0.0f || d->surface_contrast != 0.0f)
    return FALSE;
  for(int i = 0; i < LS_LIGHT_NODES; i++)
    if(d->light_node[i] != 0.0f) return FALSE;
  for(int j = 0; j < LS_HUE_NODES; j++)
    if(d->surface_hue[j] != 0.0f) return FALSE;
  return TRUE;
}

// some surface param is non-zero, so dS has to be computed (epic §M2.5)
static gboolean _surface_active(const dt_iop_lightsurface_data_t *const d)
{
  if(d->surface_contrast != 0.0f) return TRUE;
  for(int j = 0; j < LS_HUE_NODES; j++)
    if(d->surface_hue[j] != 0.0f) return TRUE;
  return FALSE;
}

// out = in * 2^g per pixel, epic §M5; alpha passes through. dS may be NULL,
// meaning 0 everywhere. at default data every term is an exact zero, so this
// returns the input bit for bit
static void _apply_gain(const float *const restrict in,
                        float *const restrict out,
                        const float *const restrict lambda,
                        const float *const restrict lambda_f,
                        const float *const restrict dS,
                        const size_t npix,
                        const dt_iop_lightsurface_data_t *const d,
                        const float m[LS_LIGHT_NODES])
{
  DT_OMP_FOR()
  for(size_t k = 0; k < npix; k++)
  {
    const float lam = lambda[k];
    const float lf = lambda_f[k];
    // §M2.4; lambda_c only meets form, which zeroes it at form = 0
    const float lc = lam - lf;
    const float dL = _light_delta(lam, lc, lf, d, m);
    const float e = exp2f(_gain(dL, dS ? dS[k] : 0.0f));
    for_three_channels(c)
      out[4 * k + c] = in[4 * k + c] * e;
    out[4 * k + 3] = in[4 * k + 3];
  }
}

/* --- which pipe does what (LS-07 Part C) --- */

// the ROI covers the whole image: this pipe can compute a layer itself.
// a downscaled high-quality export runs here at scale 1 and ends short at the
// right and bottom: imageio.c:1317 floors the output size and finalscale.c:94
// maps it back with ceil(), which drops up to about 1/s + 1 input pixels at
// output scale s. that shortfall is allowed on top of the rounding slack
static gboolean _roi_is_whole(const dt_iop_roi_t *const roi_in,
                              const dt_dev_pixelpipe_iop_t *const piece)
{
  const dt_dev_pixelpipe_t *const pipe = piece->pipe;
  const float s = pipe->processed_width > 0 && pipe->processed_height > 0
    ? fminf((float)pipe->final_width / pipe->processed_width,
            (float)pipe->final_height / pipe->processed_height)
    : 0.0f;
  const float slack = 2.0f + (s > 0.0f ? roi_in->scale / s + 1.0f : 0.0f);
  return roi_in->x == 0 && roi_in->y == 0
    && fabsf(roi_in->width - roi_in->scale * piece->buf_in.width) <= slack
    && fabsf(roi_in->height - roi_in->scale * piece->buf_in.height) <= slack;
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

// the status line, redrawn from a pipe thread. the source recomputes it from
// the module's current state rather than carrying a value, so one that fires
// after an image switch on the kept base instance shows the new image's state
static gboolean _status_idle(gpointer user_data)
{
  dt_iop_module_t *self = user_data;
  if(self->gui_data) _update_status(self);
  return G_SOURCE_REMOVE;
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
      // the status line reads "analyzing" while this runs, which can take
      // minutes; no pipe-finished signal arrives before it ends
      dt_iop_gui_enter_critical_section(self);
      g->analyzing = TRUE;
      dt_iop_gui_leave_critical_section(self);
      g_idle_add(_status_idle, self);

      dt_intrinsic_ctx_t *ctx = _get_ctx(self, model_id);
      if(ctx) L = _layer_compute(self, piece, ctx, in, roi_in);
      if(L)
      {
        L->key = key;
        L->fell_back = fell_back;
      }
      dt_iop_gui_enter_critical_section(self);
      _ls_layer_t *old = NULL;
      if(L)
      {
        old = g->layer;
        g->layer = _layer_ref(L);
      }
      g->analyzing = FALSE;
      dt_iop_gui_leave_critical_section(self);
      _layer_unref(old);
      g_idle_add(_status_idle, self);
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
    if(ctx) L = _layer_compute(self, piece, ctx, in, roi_in);
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

// the preview may have published a layer: update the status line, and redraw
// the canvases that rendered a placeholder for it. runs in the GUI thread
static void _preview_pipe_finished_callback(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  if(!g) return;

  _update_status(self);
  _update_stats(self);

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

#ifdef HAVE_AI
  // the module does not change the ROI, so in, out and the §M2 buffers share it
  assert(roi_in->width == roi_out->width && roi_in->height == roi_out->height);

  const dt_iop_lightsurface_data_t *const d = piece->data;

  // fast path, for speed only: at zero params the math below is an exact
  // identity as well (epic §M5). the gui preview still computes the layer,
  // which the light graph needs as soon as the module is enabled
  const gboolean zero = _params_are_zero(d);
  const gboolean gui_preview = self->dev->gui_attached && self->gui_data
    && dt_pipe_is_preview(piece->pipe);
  _ls_layer_t *L = zero && !gui_preview ? NULL : _layer_for_pipe(self, piece, in, roi_in);
  if(!L || zero)
  {
    dt_iop_image_copy_by_size(out, in, roi_out->width, roi_out->height, 4);
    _layer_unref(L);
    return;
  }

  const size_t npix = (size_t)roi_in->width * roi_in->height;
  const size_t nlr = (size_t)L->w * L->h;
  const gboolean surface = _surface_active(d);
  float *const guide = dt_alloc_align_float(npix * 4);
  float *const lambda = dt_alloc_align_float(npix);
  float *const lambda_f = dt_alloc_align_float(npix);
  // the layer is immutable and shared, so dS_lr belongs to this call
  float *const dS_lr = surface ? dt_alloc_align_float(nlr) : NULL;
  float *const dS = surface ? dt_alloc_align_float(npix) : NULL;
  if(guide && lambda && lambda_f && (!surface || (dS_lr && dS)))
  {
    _log_guide(in, npix, L->y_floor, guide);
    _light_full(L, guide, roi_in, piece, lambda, lambda_f);

    // surface edits are resolved at the layer's resolution, then brought to
    // the ROI like lambda (epic §M4, §M2.5)
    if(surface)
    {
      DT_OMP_FOR()
      for(size_t i = 0; i < nlr; i++)
        dS_lr[i] = _surface_delta(L->Y[i], L->rho[i], L->rho_ref, d->surface_contrast,
                                  L->hue[i], L->sat[i], d->surface_hue);
      _sample_refine(L, dS_lr, guide, roi_in, piece, dS);
    }

    // once per call, not per pixel
    float m[LS_LIGHT_NODES];
    _curve_tangents(d->light_node, m);
    _apply_gain(in, out, lambda, lambda_f, dS, npix, d, m);

    gchar *dump = _dump_prefix(piece->pipe, L->model_id);
    if(dump)
    {
      _dump(dump, "lambda_full", lambda, roi_in->width, roi_in->height, 1);
      if(dS) _dump(dump, "dS_full", dS, roi_in->width, roi_in->height, 1);
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
  else
    dt_iop_image_copy_by_size(out, in, roi_out->width, roi_out->height, 4);

  dt_free_align(guide);
  dt_free_align(lambda);
  dt_free_align(lambda_f);
  dt_free_align(dS_lr);
  dt_free_align(dS);
  _layer_unref(L);
#else
  // no AI, no light layer: the output stays an identity
  dt_iop_image_copy_by_size(out, in, roi_out->width, roi_out->height, 4);
#endif
}

/* --- GUI --- */

#ifdef HAVE_AI
// the display name of a model id, or the id itself if the list lacks it
static const char *_model_name(GList *models, const char *id)
{
  for(const GList *l = models; l; l = g_list_next(l))
  {
    const dt_intrinsic_model_t *m = l->data;
    if(!g_strcmp0(m->id, id)) return m->name;
  }
  return id;
}
#endif

// the model entries, one per installed model, plus one for an edit whose
// model is not installed
static void _rebuild_models(dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  GtkWidget *combo = g->model;

  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_combobox_clear(combo);
  g_free(g->model_hashes);
  g->model_hashes = NULL;
  g->n_models = 0;

  GString *tip = g_string_new
    (_("the AI model that separates light from surfaces.\n"
       "the choice is stored with the edit, so the edit always renders the same way.\n"
       "models differ in speed and in how they handle faces, fine shadows and textures"));

#ifdef HAVE_AI
  const dt_iop_lightsurface_params_t *p = self->params;
  const dt_iop_lightsurface_params_t *d = self->default_params;
  GList *models = dt_ai_registry_is_enabled() ? dt_intrinsic_list_models() : NULL;
  if(models)
  {
    g->n_models = g_list_length(models);
    g->model_hashes = g_new(uint32_t, g->n_models);
    g_string_append(tip, "\n");
    int selected = -1, def = 0, k = 0;
    for(const GList *l = models; l; l = g_list_next(l), k++)
    {
      const dt_intrinsic_model_t *m = l->data;
      dt_bauhaus_combobox_add(combo, m->name);
      g->model_hashes[k] = m->hash;
      if(m->hash == p->model_hash) selected = k;
      if(m->hash == d->model_hash) def = k;
      if(m->description && m->description[0])
        g_string_append_printf(tip, "\n%s: %s", m->name, m->description);
    }

    if(selected < 0)
    {
      gboolean fell_back = FALSE;
      char *id = _resolve_model_id(p->model_hash, &fell_back);
      gchar *label = g_strdup_printf(_("not installed, using %s"),
                                     id ? _model_name(models, id) : _("none"));
      dt_bauhaus_combobox_add(combo, label);
      g_free(label);
      g_free(id);
      selected = g->n_models;
    }
    dt_bauhaus_combobox_set_default(combo, def);
    dt_bauhaus_combobox_set(combo, selected);
  }
  dt_intrinsic_models_free(models);
#endif

  if(!g->n_models)
  {
    dt_bauhaus_combobox_add(combo, _("none"));
    dt_bauhaus_combobox_set_default(combo, 0);
    dt_bauhaus_combobox_set(combo, 0);
  }
  gtk_widget_set_sensitive(combo, g->n_models > 0);
  gtk_widget_set_tooltip_text(combo, tip->str);
  g_string_free(tip, TRUE);
  DT_LEAVE_GUI_UPDATE();
}

// the status line under the model, and the relief slider, which only models
// that separate fine shading from the broad light use
static void _update_status(dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  const dt_iop_lightsurface_params_t *p = self->params;
  gboolean relief_shown = gtk_widget_get_visible(g->relief);
  // only a layer of the current model tells whether it uses relief; until
  // one arrives the visibility is the previous model's, so no note
  gboolean relief_known = FALSE;
  gchar *text = NULL;

#ifdef HAVE_AI
  const gboolean enabled = dt_ai_registry_is_enabled();
  GList *models = enabled ? dt_intrinsic_list_models() : NULL;
  gboolean fell_back = FALSE;
  char *id = models ? _resolve_model_id(p->model_hash, &fell_back) : NULL;

  if(!enabled)
    text = g_strdup(_("AI is disabled in preferences > AI: this module has no effect"));
  else if(!id)
    text = g_strdup(_("no intrinsic decomposition model installed: this module has no effect"));
  else if(_ctx_failed(self, id))
    text = g_strdup_printf(_("model %s failed to load, see the log (-d ai)"),
                           _model_name(models, id));
  else
  {
    // the layer's upstream hash is not compared: the preview flags a
    // recomputation for a changed upstream itself (g->analyzing)
    const uint32_t hash = dt_intrinsic_hash(id);
    dt_iop_gui_enter_critical_section(self);
    const _ls_layer_t *L = g->layer;
    const gboolean analyzing = g->analyzing;
    const gboolean ready = L && !analyzing
      && L->key.imgid == self->dev->image_storage.id
      && L->key.model_hash == hash;
    const double ms = ready ? L->ms : 0.0;
    const gboolean relief = ready ? L->mp.relief : TRUE;
    dt_iop_gui_leave_critical_section(self);

    if(ready)
    {
      relief_shown = relief;
      relief_known = TRUE;
      // the edit's own model_hash decides, not the layer's flag: a layer
      // computed as a fallback is reused once the edit chooses that model
      if(fell_back)
        text = g_strdup_printf(_("this edit's model is not installed, using %s"),
                               _model_name(models, id));
      else
        text = g_strdup_printf(_("analysis %.1f s"), ms / 1000.0);
    }
    else if(analyzing || self->enabled)
      text = g_strdup(_("analyzing the image..."));
    else
      text = g_strdup(""); // a disabled module analyzes nothing
  }
  g_free(id);
  dt_intrinsic_models_free(models);
#else
  text = g_strdup(_("built without AI support: this module has no effect"));
#endif

  gtk_widget_set_visible(g->relief, relief_shown);
  // a value set under another model stays: hiding the slider must not reset it
  if(relief_known && !relief_shown && p->relief != 0.0f)
  {
    gchar *t = g_strconcat(text, _(" (relief is not used by this model)"), NULL);
    g_free(text);
    text = t;
  }
  gtk_label_set_text(GTK_LABEL(g->status_label), text);
  g_free(text);
}

static void _model_changed(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  dt_iop_lightsurface_params_t *p = self->params;
  const int pos = dt_bauhaus_combobox_get(widget);
  // the "not installed" entry is the edit's current state, not a choice
  if(pos < 0 || pos >= g->n_models || p->model_hash == g->model_hashes[pos]) return;

  p->model_hash = g->model_hashes[pos];
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  _rebuild_models(self);
  _update_status(self);
  _update_stats(self);
}

static void _display_changed(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  const int mode = CLAMP(dt_bauhaus_combobox_get(widget), LS_DISPLAY_NORMAL, LS_DISPLAY_CHANGES);
  dt_iop_gui_enter_critical_section(self);
  g->display_mode = mode;
  dt_iop_gui_leave_critical_section(self);
  dt_iop_refresh_center(self);
}

static void _warn_toggled(GtkGestureSingle *gesture,
                          int n_press,
                          double x,
                          double y,
                          dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_request_focus(self);

  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  dt_iop_gui_enter_critical_section(self);
  const gboolean show = g->show_warn = !g->show_warn;
  dt_iop_gui_leave_critical_section(self);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->warn_toggle), show);
  dt_iop_refresh_center(self);
}

// epic §M3.2 applied from the node the user moved outwards, in both
// directions, so that no slider shows a value commit_params() would change.
// it only pushes: moving the node back leaves its neighbors where they are
static void _push_nodes(dt_iop_module_t *self, const int i)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  float *const v = ((dt_iop_lightsurface_params_t *)self->params)->light_node;
  const float step = 1.0f - LS_NODE_MIN_STEP;

  DT_ENTER_GUI_UPDATE();
  for(int k = i + 1; k < LS_LIGHT_NODES; k++)
    if(v[k] < v[k - 1] - step)
    {
      v[k] = v[k - 1] - step;
      dt_bauhaus_slider_set(g->light_node[k], v[k]);
    }
  for(int k = i - 1; k >= 0; k--)
    if(v[k] > v[k + 1] + step)
    {
      v[k] = v[k + 1] + step;
      dt_bauhaus_slider_set(g->light_node[k], v[k]);
    }
  DT_LEAVE_GUI_UPDATE();
}

/* --- light graph --- */

// the node values as commit_params() sanitizes them, and their tangents
static void _graph_nodes(const dt_iop_lightsurface_params_t *const p,
                         dt_iop_lightsurface_data_t *const d,
                         float m[LS_LIGHT_NODES])
{
  memcpy(d, p, sizeof(*d));
  _sanitize_nodes(d->light_node);
  _curve_tangents(d->light_node, m);
}

#ifdef HAVE_AI
static inline void _hist_add(float hist[LS_HIST_BINS], const float v)
{
  const float b = (v - LS_HIST_MIN) * (LS_HIST_BINS / (LS_HIST_MAX - LS_HIST_MIN));
  // also false for NaN
  if(b >= 0.0f && b < (float)LS_HIST_BINS) hist[(int)b] += 1.0f;
}

// the k-th smallest of a[0, n), which ends up at a[k] with nothing larger
// before it (Hoare's selection). the stats run on every drag step, where
// two full sorts of a layer took 64 ms
static float _select(float *const a, const int64_t n, const int64_t k)
{
  int64_t lo = 0, hi = n - 1;
  while(lo < hi)
  {
    const float pivot = a[lo + (hi - lo) / 2];
    int64_t i = lo, j = hi;
    while(i <= j)
    {
      while(a[i] < pivot) i++;
      while(a[j] > pivot) j--;
      if(i <= j)
      {
        const float t = a[i];
        a[i++] = a[j];
        a[j--] = t;
      }
    }
    // now a[lo, j] <= pivot <= a[i, hi], and anything between equals pivot
    if(k <= j)
      hi = j;
    else if(k >= i)
      lo = i;
    else
      break;
  }
  return a[k];
}

// P90 - P10 of the finite values of a, nearest rank as in _percentile();
// reorders a. 0 if there are none
static float _lighting_ratio(float *const a, const size_t n)
{
  size_t m = 0;
  for(size_t k = 0; k < n; k++)
    if(dt_isfinite(a[k])) a[m++] = a[k];
  if(!m) return 0.0f;
  const int64_t k90 = (int64_t)floor(90.0 / 100.0 * (double)(m - 1));
  const int64_t k10 = (int64_t)floor(10.0 / 100.0 * (double)(m - 1));
  const float p90 = _select(a, m, k90);
  // the selection left the k90 smallest values in front of a[k90]
  const float p10 = k10 < k90 ? _select(a, k90, k10) : p90;
  return p90 - p10;
}

// epic §M6 for a layer and the committed params d (nodes sanitized, m their
// tangents): histograms of the light level before and after the light edit,
// lambda and lambda + dL, counted on every 2nd pixel in x and y, and the
// lighting ratio of each, taken on the same samples. FALSE if out of memory
static gboolean _light_stats(const _ls_layer_t *const L,
                             const dt_iop_lightsurface_data_t *const d,
                             const float m[LS_LIGHT_NODES],
                             float hist_before[LS_HIST_BINS],
                             float hist_after[LS_HIST_BINS],
                             float *const ratio_before,
                             float *const ratio_after)
{
  const int sw = (L->w + 1) / 2;
  const int sh = (L->h + 1) / 2;
  const size_t ns = (size_t)sw * sh;
  float *const before = dt_alloc_align_float(ns);
  float *const after = dt_alloc_align_float(ns);
  if(!before || !after)
  {
    dt_free_align(before);
    dt_free_align(after);
    return FALSE;
  }

  DT_OMP_FOR()
  for(size_t s = 0; s < ns; s++)
  {
    const size_t k = (size_t)2 * (s / sw) * L->w + 2 * (s % sw);
    const float lambda = L->lambda[k];
    before[s] = lambda;
    after[s] = lambda + _light_delta(lambda, L->lambda_c[k], L->lambda_f[k], d, m);
  }

  memset(hist_before, 0, sizeof(float) * LS_HIST_BINS);
  memset(hist_after, 0, sizeof(float) * LS_HIST_BINS);
  for(size_t s = 0; s < ns; s++)
  {
    _hist_add(hist_before, before[s]);
    _hist_add(hist_after, after[s]);
  }
  *ratio_before = _lighting_ratio(before, ns);
  *ratio_after = _lighting_ratio(after, ns);

  dt_free_align(before);
  dt_free_align(after);
  return TRUE;
}
#endif

// the graph's statistics and the ratio label, from the current layer
static void _update_stats(dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  gboolean valid = FALSE;

#ifdef HAVE_AI
  const dt_iop_lightsurface_params_t *p = self->params;
  // only a layer of this image and of the model the edit uses, as for the
  // status line: until the right one arrives the graph shows no histogram
  gboolean fell_back = FALSE;
  char *id = dt_ai_registry_is_enabled() ? _resolve_model_id(p->model_hash, &fell_back) : NULL;
  const uint32_t hash = id ? dt_intrinsic_hash(id) : 0;
  g_free(id);

  dt_iop_gui_enter_critical_section(self);
  _ls_layer_t *L = hash && g->layer
    && g->layer->key.imgid == self->dev->image_storage.id
    && g->layer->key.model_hash == hash
    ? _layer_ref(g->layer) : NULL;
  dt_iop_gui_leave_critical_section(self);

  if(L)
  {
    dt_iop_lightsurface_data_t d;
    float m[LS_LIGHT_NODES];
    _graph_nodes(p, &d, m);
    valid = _light_stats(L, &d, m, g->hist_before, g->hist_after,
                         &g->ratio_before, &g->ratio_after);
    _layer_unref(L);
  }
#endif

  g->hist_valid = valid;
  gchar *text = NULL;
  if(!valid)
    text = g_strdup("");
  else if(fabsf(g->ratio_after - g->ratio_before) < 0.05f)
    text = g_strdup_printf(_("lighting ratio %.1f EV"), g->ratio_before);
  else
    text = g_strdup_printf(_("lighting ratio %.1f EV → %.1f EV"),
                           g->ratio_before, g->ratio_after);
  gtk_label_set_text(GTK_LABEL(g->ratio_label), text);
  g_free(text);
  gtk_widget_queue_draw(GTK_WIDGET(g->area));
}

static void _graph_geometry(GtkWidget *widget, int *inset, int *width, int *height)
{
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  *inset = DT_PIXEL_APPLY_DPI(4);
  *width = allocation.width - 2 * (*inset);
  *height = allocation.height - 2 * (*inset) - DT_RESIZE_HANDLE_SIZE;
}

// light level in EV to graph x, and back
static inline float _graph_x(const float ev, const int width)
{
  return (ev - LS_GRAPH_X_MIN) / (LS_GRAPH_X_MAX - LS_GRAPH_X_MIN) * width;
}

static inline float _graph_ev(const double x, const int width)
{
  return LS_GRAPH_X_MIN + (float)x / MAX(width, 1) * (LS_GRAPH_X_MAX - LS_GRAPH_X_MIN);
}

// correction in EV to graph y, and back
static inline float _graph_y(const float c, const int height)
{
  return (0.5f - 0.5f * c / LS_GRAPH_Y_MAX) * height;
}

static inline float _graph_c(const double y, const int height)
{
  return (1.0f - 2.0f * (float)y / MAX(height, 1)) * LS_GRAPH_Y_MAX;
}

// x_i of light node i: -7 ... +1 EV
static inline float _node_ev(const int i)
{
  return (float)(i - (LS_LIGHT_NODES - 2));
}

// the node drawn within 8 px of (x, y) in graph coordinates, the nearest
// if several are; -1 if none is
static int _node_at(dt_iop_module_t *self,
                    const double x, const double y,
                    const int width, const int height)
{
  dt_iop_lightsurface_data_t d;
  float m[LS_LIGHT_NODES];
  _graph_nodes(self->params, &d, m);

  const double r = DT_PIXEL_APPLY_DPI(8);
  double best = r * r;
  int node = -1;
  for(int i = 0; i < LS_LIGHT_NODES; i++)
  {
    const double dx = x - _graph_x(_node_ev(i), width);
    const double dy = y - _graph_y(d.light_node[i], height);
    const double d2 = dx * dx + dy * dy;
    if(d2 <= best)
    {
      best = d2;
      node = i;
    }
  }
  return node;
}

// a light histogram as a path through its bin centers, from and back to
// the bottom of the graph
static void _hist_path(cairo_t *cr,
                       const float hist[LS_HIST_BINS],
                       const float scale,
                       const int width, const int height)
{
  const float bin = (LS_HIST_MAX - LS_HIST_MIN) / LS_HIST_BINS;
  cairo_move_to(cr, _graph_x(LS_HIST_MIN + 0.5f * bin, width), height);
  for(int b = 0; b < LS_HIST_BINS; b++)
    cairo_line_to(cr, _graph_x(LS_HIST_MIN + (b + 0.5f) * bin, width),
                  height - scale * hist[b]);
  cairo_line_to(cr, _graph_x(LS_HIST_MAX - 0.5f * bin, width), height);
}

static gboolean _area_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;

  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  if(width <= 0 || height <= 0) return FALSE;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  cairo_surface_t *cst =
    dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, allocation.width, allocation.height);
  cairo_t *cr = cairo_create(cst);
  gtk_render_background(gtk_widget_get_style_context(widget), cr,
                        0, 0, allocation.width, allocation.height);
  cairo_translate(cr, inset, inset);

  // background and border
  cairo_rectangle(cr, 0, 0, width, height);
  set_color(cr, darktable.bauhaus->graph_bg);
  cairo_fill_preserve(cr);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
  set_color(cr, darktable.bauhaus->graph_border);
  cairo_stroke(cr);

  cairo_save(cr);
  cairo_rectangle(cr, 0, 0, width, height);
  cairo_clip(cr);

  // a grid line per EV of light, and the zero correction
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));
  set_color(cr, darktable.bauhaus->graph_grid);
  for(int ev = (int)ceilf(LS_GRAPH_X_MIN); ev <= (int)floorf(LS_GRAPH_X_MAX); ev++)
  {
    if(ev == 0) continue;
    const float x = _graph_x(ev, width);
    dt_draw_line(cr, x, 0, x, height);
  }
  cairo_stroke(cr);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
  dt_draw_line(cr, 0, _graph_y(0.0f, height), width, _graph_y(0.0f, height));
  cairo_stroke(cr);

  // the key light
  const float key_x = _graph_x(0.0f, width);
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
  dt_draw_line(cr, key_x, 0, key_x, height);
  cairo_stroke(cr);

  // the histograms, scaled together so that the visible part of "before"
  // peaks at 90 % of the height: they measure image area, not correction
  if(g->hist_valid)
  {
    const float bin = (LS_HIST_MAX - LS_HIST_MIN) / LS_HIST_BINS;
    float peak = 0.0f;
    for(int b = 0; b < LS_HIST_BINS; b++)
      if(LS_HIST_MIN + (b + 0.5f) * bin >= LS_GRAPH_X_MIN)
        peak = fmaxf(peak, g->hist_before[b]);
    if(peak > 0.0f)
    {
      const float scale = 0.9f * height / peak;
      const GdkRGBA fg = darktable.bauhaus->graph_fg;
      _hist_path(cr, g->hist_before, scale, width, height);
      cairo_close_path(cr);
      cairo_set_source_rgba(cr, fg.red, fg.green, fg.blue, 0.3);
      cairo_fill(cr);
      _hist_path(cr, g->hist_after, scale, width, height);
      cairo_set_source_rgba(cr, fg.red, fg.green, fg.blue, 0.8);
      cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
      cairo_stroke(cr);
    }
  }

  dt_iop_lightsurface_data_t d;
  float m[LS_LIGHT_NODES];
  _graph_nodes(self->params, &d, m);

  // the node curve's correction C(x)
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.0));
  for(int px = 0; px <= width; px++)
  {
    const float y = _graph_y(_light_correction(_graph_ev(px, width), d.light_node, m), height);
    if(px)
      cairo_line_to(cr, px, y);
    else
      cairo_move_to(cr, px, y);
  }
  cairo_stroke(cr);

  // with fill, the total correction C(x) + a_3(x + C(x)): the light delta
  // of a pixel with no form or relief to scale
  if(d.fill > 0.0f)
  {
    const double dashes[2] = { DT_PIXEL_APPLY_DPI(4.0), DT_PIXEL_APPLY_DPI(3.0) };
    cairo_set_dash(cr, dashes, 2, 0.0);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
    for(int px = 0; px <= width; px++)
    {
      const float y = _graph_y(_light_delta(_graph_ev(px, width), 0.0f, 0.0f, &d, m), height);
      if(px)
        cairo_line_to(cr, px, y);
      else
        cairo_move_to(cr, px, y);
    }
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
  }

  // the nodes
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
  for(int i = 0; i < LS_LIGHT_NODES; i++)
  {
    cairo_arc(cr, _graph_x(_node_ev(i), width), _graph_y(d.light_node[i], height),
              DT_PIXEL_APPLY_DPI(4.0), 0.0, 2.0 * M_PI);
    set_color(cr, darktable.bauhaus->graph_fg);
    cairo_stroke_preserve(cr);
    if(i == g->hovered_node || i == g->dragged_node)
      set_color(cr, darktable.bauhaus->graph_fg);
    else
      set_color(cr, darktable.bauhaus->graph_bg);
    cairo_fill(cr);
  }

  // the light level under the pointer on the image
  if(g->cursor_lambda_valid)
  {
    const float x = _graph_x(g->cursor_lambda, width);
    set_color(cr, darktable.bauhaus->graph_fg_active);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
    dt_draw_line(cr, x, 0, x, height);
    cairo_stroke(cr);
  }

  // the key light's label, at the bottom right of its line
  PangoFontDescription *desc = dt_gui_get_font();
  PangoLayout *layout = pango_cairo_create_layout(cr);
  pango_layout_set_font_description(layout, desc);
  pango_cairo_context_set_resolution(pango_layout_get_context(layout), darktable.gui->dpi);
  pango_layout_set_text(layout, C_("light graph", "key"), -1);
  PangoRectangle ink;
  pango_layout_get_pixel_extents(layout, &ink, NULL);
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_move_to(cr, key_x + DT_PIXEL_APPLY_DPI(3) - ink.x,
                height - DT_PIXEL_APPLY_DPI(3) - ink.height - ink.y);
  pango_cairo_show_layout(cr, layout);
  g_object_unref(layout);
  pango_font_description_free(desc);

  cairo_restore(cr);
  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return FALSE;
}

// sets light node i from the graph, as moving its slider would: the
// neighbors are pushed (epic §M3.2) and one history item is added
static void _graph_set_node(dt_iop_module_t *self, const int i, const float v)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  dt_iop_lightsurface_params_t *p = self->params;
  const float value = CLAMP(v, -LS_GRAPH_Y_MAX, LS_GRAPH_Y_MAX);
  if(p->light_node[i] == value) return;

  p->light_node[i] = value;
  _push_nodes(self, i);
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_slider_set(g->light_node[i], value);
  DT_LEAVE_GUI_UPDATE();
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  _update_stats(self);
}

static void _area_motion(GtkEventControllerMotion *controller,
                         double x, double y,
                         dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  GtkWidget *widget = dt_gui_get_widget(controller);
  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  const double gx = x - inset, gy = y - inset;
  g->pointer_in = TRUE;
  g->pointer_x = gx;

  // a release the gesture never reported (a broken grab) must not leave
  // the node following the pointer
  if(g->dragged_node >= 0 && !(dt_key_modifier_state() & GDK_BUTTON1_MASK))
    g->dragged_node = -1;

  if(g->dragged_node >= 0)
  {
    _graph_set_node(self, g->dragged_node, _graph_c(gy, height) + g->drag_offset);
    return;
  }

  const int node = _node_at(self, gx, gy, width, height);
  if(node != g->hovered_node)
  {
    g->hovered_node = node;
    gtk_widget_queue_draw(widget);
  }
}

static void _area_leave(GtkEventControllerMotion *controller, dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  g->pointer_in = FALSE;
  if(g->dragged_node >= 0) return;
  g->hovered_node = -1;
  gtk_widget_queue_draw(dt_gui_get_widget(controller));
}

static void _area_button_press(GtkGestureSingle *gesture,
                               int n_press,
                               double x, double y,
                               dt_iop_module_t *self)
{
  if(gtk_gesture_single_get_current_button(gesture) != GDK_BUTTON_PRIMARY) return;

  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  GtkWidget *widget = dt_gui_get_widget(gesture);
  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  const int node = _node_at(self, x - inset, y - inset, width, height);
  if(node < 0) return;

  g->hovered_node = node;
  if(n_press == 2)
  {
    // the first press started a drag; this one resets the node instead
    g->dragged_node = -1;
    _graph_set_node(self, node, 0.0f);
  }
  else
  {
    // the node keeps its distance to the pointer, so grabbing it off
    // center does not move it
    dt_iop_lightsurface_data_t d;
    float m[LS_LIGHT_NODES];
    _graph_nodes(self->params, &d, m);
    g->dragged_node = node;
    g->drag_offset = d.light_node[node] - _graph_c(y - inset, height);
  }
  gtk_widget_queue_draw(widget);
}

static void _area_button_release(GtkGestureSingle *gesture,
                                 int n_press,
                                 double x, double y,
                                 dt_iop_module_t *self)
{
  if(gtk_gesture_single_get_current_button(gesture) != GDK_BUTTON_PRIMARY) return;

  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  g->dragged_node = -1;
  if(!g->pointer_in) g->hovered_node = -1;
  gtk_widget_queue_draw(dt_gui_get_widget(gesture));
}

// the node nearest the pointer's light level, in tone equalizer's steps
// (toneequal.c:2908-2913)
static void _area_scrolled(GtkEventControllerScroll *controller,
                           double dx, double dy,
                           dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  if(dy == 0.0 || !g->pointer_in) return;

  const GdkModifierType state = dt_key_modifier_state();
  float step;
  if(dt_modifier_is(state, GDK_SHIFT_MASK))
    step = 1.0f;  // coarse
  else if(dt_modifier_is(state, GDK_CONTROL_MASK))
    step = 0.1f;  // fine
  else if(dt_modifier_is(state, 0))
    step = 0.25f; // standard
  else
    return;       // shift+alt resizes the graph (dt_ui_resize_wrap())

  int inset, width, height;
  _graph_geometry(dt_gui_get_widget(controller), &inset, &width, &height);
  const float ev = _graph_ev(g->pointer_x, width);
  const int i = CLAMP((int)roundf(ev - _node_ev(0)), 0, LS_LIGHT_NODES - 1);
  const dt_iop_lightsurface_params_t *p = self->params;
  _graph_set_node(self, i, p->light_node[i] - (float)dy * step);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;

  for(int i = 0; i < LS_LIGHT_NODES; i++)
    if(w == g->light_node[i])
    {
      _push_nodes(self, i);
      break;
    }

  if(!w || w == g->relief) _update_status(self);
  // gui_update() comes through here too
  _update_stats(self);
}

void gui_update(dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  _rebuild_models(self);
  dt_gui_update_collapsible_section(&g->nodes_section);
  gui_changed(self, NULL, NULL);
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  if(in || !g) return;

  // as toneequal.c does with its mask: leaving the module shows the image again
  dt_iop_gui_enter_critical_section(self);
  const gboolean was_shown = g->display_mode != LS_DISPLAY_NORMAL || g->show_warn;
  g->display_mode = LS_DISPLAY_NORMAL;
  g->show_warn = FALSE;
  dt_iop_gui_leave_critical_section(self);

  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_combobox_set(g->display, LS_DISPLAY_NORMAL);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->warn_toggle), FALSE);
  DT_LEAVE_GUI_UPDATE();

  if(was_shown) dt_iop_refresh_center(self);
}

// sliders.md case 2, in its order: the digits come last
static void _slider_percent(GtkWidget *w)
{
  dt_bauhaus_slider_set_factor(w, 100.0f);
  dt_bauhaus_slider_set_format(w, "%");
  dt_bauhaus_slider_set_digits(w, 1);
}

void gui_init(dt_iop_module_t *self)
{
  static const char *const hue_labels[LS_HUE_NODES] =
    { N_("red"), N_("orange"), N_("yellow"), N_("green"),
      N_("cyan"), N_("blue"), N_("lavender"), N_("magenta") };

  dt_iop_lightsurface_gui_data_t *g = IOP_GUI_ALLOC(lightsurface);

  static dt_action_def_t notebook_def = { };
  g->notebook = dt_ui_notebook_new(&notebook_def);
  dt_action_define_iop(self, NULL, N_("page"), GTK_WIDGET(g->notebook), &notebook_def);

  /* light */
  GtkWidget *page = self->widget = dt_ui_notebook_page(g->notebook, N_("light"), NULL);

  g->hovered_node = g->dragged_node = -1;
  g->area = GTK_DRAWING_AREA(dt_ui_resize_wrap
                             (NULL, 0, "plugins/darkroom/lightsurface/graphheight"));
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_area_draw), self);
  dt_gui_connect_click(g->area, _area_button_press, _area_button_release, self);
  dt_gui_connect_motion(g->area, _area_motion, _area_motion, _area_leave, self);
  dt_gui_connect_scroll(g->area, GTK_EVENT_CONTROLLER_SCROLL_BOTH_AXES
                                 | GTK_EVENT_CONTROLLER_SCROLL_DISCRETE, _area_scrolled, self);
  gtk_widget_set_tooltip_text
    (GTK_WIDGET(g->area),
     _("how much of the image is lit at each level, relative to the key light (0 EV):"
       " filled before the edit, outlined after it.\n"
       "the curve is the correction applied at each level.\n"
       "drag a node to move it, scroll to move the nearest one, double-click to reset it"));
  dt_gui_box_add(page, g->area);

  g->ratio_label = dt_ui_label_new("");
  dt_gui_box_add(page, g->ratio_label);

  g->form = dt_bauhaus_slider_from_params(self, "form");
  _slider_percent(g->form);
  dt_bauhaus_slider_set_soft_range(g->form, -0.9f, 2.0f);
  gtk_widget_set_tooltip_text
    (g->form,
     _("strength of the modeling light: how much the light changes across shapes"
       " and between sun and shade.\n"
       "positive: harder, more dramatic light. negative: softer light.\n"
       "surfaces (materials) are not affected"));

  g->relief = dt_bauhaus_slider_from_params(self, "relief");
  _slider_percent(g->relief);
  gtk_widget_set_tooltip_text
    (g->relief,
     _("contrast of the fine shading inside textures such as bark, stone or fabric,"
       " without changing the materials' own tones"));
  // _update_status() hides it for models without a relief layer
  gtk_widget_set_no_show_all(g->relief, TRUE);
  gtk_widget_set_visible(g->relief, TRUE);

  g->fill = dt_bauhaus_slider_from_params(self, "fill");
  _slider_percent(g->fill);
  dt_bauhaus_slider_set_soft_range(g->fill, 0.0f, 0.5f);
  gtk_widget_set_tooltip_text
    (g->fill,
     /* xgettext:no-c-format */
     _("adds light like a fill light would: in proportion to the key light,"
       " so deep shade is lifted a lot and sunlit areas barely change.\n"
       "25 % is a fill light 2 EV below the key light"));

  dt_gui_new_collapsible_section(&g->nodes_section, "plugins/darkroom/lightsurface/expand_nodes",
                                 _("light curve nodes"), GTK_BOX(page), DT_ACTION(self));
  self->widget = GTK_WIDGET(g->nodes_section.container);
  for(int i = 0; i < LS_LIGHT_NODES; i++)
  {
    gchar *field = g_strdup_printf("light_node[%d]", i);
    g->light_node[i] = dt_bauhaus_slider_from_params(self, field);
    g_free(field);
    // the node's EV offset from the key light, as in "-7 EV", "0 EV", "+1 EV"
    const int ev = i - (LS_LIGHT_NODES - 2);
    gchar *label = g_strdup_printf(ev ? "%+d EV" : "%d EV", ev);
    dt_bauhaus_widget_set_label(g->light_node[i], NULL, label);
    g_free(label);
    dt_bauhaus_slider_set_format(g->light_node[i], _(" EV"));
    dt_bauhaus_slider_set_digits(g->light_node[i], 2);
    gtk_widget_set_tooltip_text(g->light_node[i],
                                _("correction applied to areas lit at this level,"
                                  " relative to the key light (0 EV)"));
  }

  /* surface */
  page = self->widget = dt_ui_notebook_page(g->notebook, N_("surface"), NULL);

  g->surface_contrast = dt_bauhaus_slider_from_params(self, "surface_contrast");
  _slider_percent(g->surface_contrast);
  gtk_widget_set_tooltip_text
    (g->surface_contrast,
     _("contrast between materials, e.g. calms a patchy wall or blotchy skin,"
       " without changing the light on them"));

  dt_gui_box_add(page, dt_ui_section_label_new(_("lightness by hue")));
  for(int j = 0; j < LS_HUE_NODES; j++)
  {
    gchar *field = g_strdup_printf("surface_hue[%d]", j);
    g->surface_hue[j] = dt_bauhaus_slider_from_params(self, field);
    g_free(field);
    dt_bauhaus_widget_set_label(g->surface_hue[j], NULL, hue_labels[j]);
    dt_bauhaus_slider_set_format(g->surface_hue[j], _(" EV"));
    dt_bauhaus_slider_set_digits(g->surface_hue[j], 2);
    gtk_widget_set_tooltip_text(g->surface_hue[j],
                                _("lightens or darkens materials of this color, weighted"
                                  " by how saturated they are.\n"
                                  "shadows on them are not deepened"));

    // a dark to light ramp of the node's hue. the tint is only a cue: it
    // uses the conventional HSL hue at j * 45 degrees, while the module
    // places its nodes on the UCS hue angle of _hue_sat(), which differs a
    // little
    for(int s = 0; s < 16; s++)
    {
      const float stop = s / 15.0f;
      dt_aligned_pixel_t rgb;
      hsl2rgb(rgb, j / (float)LS_HUE_NODES, 0.6f, 0.15f + 0.7f * stop);
      dt_bauhaus_slider_set_stop(g->surface_hue[j], stop, rgb[0], rgb[1], rgb[2]);
    }
  }

  /* view */
  page = self->widget = dt_ui_notebook_page(g->notebook, N_("view"), NULL);

  // built by hand: its entries are the installed models (_rebuild_models())
  g->model = dt_bauhaus_combobox_new_full(DT_ACTION(self), NULL, N_("model"), NULL, 0,
                                          (GtkCallback)_model_changed, self, NULL);
  dt_gui_box_add(page, g->model);

  DT_BAUHAUS_COMBOBOX_NEW_FULL(g->display, self, NULL, N_("display"), NULL,
                               LS_DISPLAY_NORMAL, _display_changed, self,
                               N_("normal"), N_("light layer"), N_("surface layer"),
                               N_("changes"));
  dt_gui_box_add(page, g->display);

  g->warn_toggle = dt_iop_togglebutton_new(self, NULL, N_("show unreliable areas"), NULL,
                                           G_CALLBACK(_warn_toggled), FALSE, 0, 0,
                                           dtgtk_cairo_paint_showmask, NULL);
  dt_gui_add_class(g->warn_toggle, "dt_transparent_background");
  dt_gui_add_class(g->warn_toggle, "dt_bauhaus_alignment");
  dt_gui_box_add(page, dt_gui_hbox(dt_gui_expand(dt_ui_label_new(_("show unreliable areas"))),
                                   g->warn_toggle));

  g->status_label = dt_ui_label_new("");
  gtk_label_set_ellipsize(GTK_LABEL(g->status_label), PANGO_ELLIPSIZE_NONE);
  gtk_label_set_line_wrap(GTK_LABEL(g->status_label), TRUE);
  dt_gui_box_add(page, g->status_label);

  // the light x surface map goes here (LS-13)
  g->map_box = dt_gui_vbox();
  dt_gui_box_add(page, g->map_box);

  self->widget = GTK_WIDGET(g->notebook);

#ifdef HAVE_AI
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED,
                           _preview_pipe_finished_callback);
#endif
}

// the signal is disconnected by dt_iop_gui_cleanup_module() before this runs
// (DT_CONTROL_SIGNAL_DISCONNECT_ALL, src/develop/imageop.c:2470)
void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_lightsurface_gui_data_t *g = self->gui_data;
  g_free(g->model_hashes);
  g->model_hashes = NULL;
#ifdef HAVE_AI
  // teardown holds the screen pipes' locks, so nothing queues a new one
  // after this drain (dev-doc/GUI_Threading.md)
  while(g_idle_remove_by_data(self)) ;

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
