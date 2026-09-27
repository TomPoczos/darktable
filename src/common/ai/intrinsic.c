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

#include "common/ai/intrinsic.h"
#include "ai/backend.h"
#include "common/ai_models.h"
#include "common/darktable.h"
#include "common/math.h"
#include <stdlib.h>
#include <string.h>

#define TASK_KEY "intrinsic"

// floor for the returned shading, the epic's eps (§M0)
#define SHADING_FLOOR 0x1p-20f

// the "arch" values of the models in the LS-03 decision record
// (tasks/light-and-surface/epic-01/evidence/03-go-no-go.md)
typedef enum dt_intrinsic_arch_t
{
  DT_INTRINSIC_ARCH_MARIGOLD,  // "marigold": Marigold-IID-Lighting v1.1
  DT_INTRINSIC_ARCH_COMPPHOTO  // "compphoto": compphoto/Intrinsic, ordinal shading
} dt_intrinsic_arch_t;

struct dt_intrinsic_env_t
{
  dt_ai_environment_t *ai_env;
};

struct dt_intrinsic_ctx_t
{
  dt_ai_context_t *ai_ctx;
  dt_intrinsic_arch_t arch;
  char *model_id;
  uint32_t hash;
  dt_intrinsic_model_params_t params;

  // output shape from the graph; a dim <= 0 is symbolic
  int64_t out_shape[4];
  gboolean out_dynamic;

  GMutex lock;
};

/* --- listing --- */

// the listing environment is created on first use and kept for the whole
// session. it is not a g_once: dt_ai_env_init() returns NULL while AI is
// disabled, and a NULL cached by g_once would hide the models for the rest
// of the session after the user enables AI
static GMutex _list_lock;
static dt_ai_environment_t *_list_env = NULL;

static dt_ai_environment_t *_listing_env(void)
{
  g_mutex_lock(&_list_lock);
  if(!_list_env)
    _list_env = dt_ai_env_init(NULL);
  dt_ai_environment_t *env = _list_env;
  g_mutex_unlock(&_list_lock);
  return env;
}

uint32_t dt_intrinsic_hash(const char *model_id)
{
  // copy of spektrafilm's _name_hash() (src/iop/spektrafilm.c:744)
  const dt_hash_t h = dt_hash(DT_INITHASH, model_id, strlen(model_id));
  const uint32_t h32 = (uint32_t)(h ^ (h >> 32));
  return h32 ? h32 : 1; // 0 is reserved for "none"
}

static gint _model_name_cmp(gconstpointer a, gconstpointer b)
{
  const dt_intrinsic_model_t *ma = a;
  const dt_intrinsic_model_t *mb = b;
  return g_strcmp0(ma->name, mb->name);
}

GList *dt_intrinsic_list_models(void)
{
  dt_ai_environment_t *env = _listing_env();
  if(!env) return NULL;

  GList *models = NULL;
  const int count = dt_ai_get_model_count(env);
  for(int i = 0; i < count; i++)
  {
    const dt_ai_model_info_t *info = dt_ai_get_model_info_by_index(env, i);
    if(!info || !info->id || g_strcmp0(info->task_type, TASK_KEY)) continue;

    dt_intrinsic_model_t *m = g_new0(dt_intrinsic_model_t, 1);
    m->id = g_strdup(info->id);
    m->name = g_strdup(info->name ? info->name : info->id);
    m->hash = dt_intrinsic_hash(info->id);
    models = g_list_prepend(models, m);
  }
  return g_list_sort(models, _model_name_cmp);
}

static void _model_free(gpointer data)
{
  dt_intrinsic_model_t *m = data;
  g_free(m->id);
  g_free(m->name);
  g_free(m);
}

void dt_intrinsic_models_free(GList *models)
{
  g_list_free_full(models, _model_free);
}

gboolean dt_intrinsic_available(void)
{
  if(!dt_ai_registry_is_enabled()) return FALSE;

  GList *models = dt_intrinsic_list_models();
  const gboolean available = models != NULL;
  dt_intrinsic_models_free(models);
  return available;
}

static gboolean _is_discoverable(const char *model_id)
{
  GList *models = dt_intrinsic_list_models();
  gboolean found = FALSE;
  for(const GList *l = models; l && !found; l = g_list_next(l))
    found = !g_strcmp0(((dt_intrinsic_model_t *)l->data)->id, model_id);
  dt_intrinsic_models_free(models);
  return found;
}

char *dt_intrinsic_active_model_id(void)
{
  char *id = dt_ai_models_get_active_for_task(TASK_KEY);
  if(!id || !id[0] || !_is_discoverable(id))
  {
    g_free(id);
    return NULL;
  }
  return id;
}

char *dt_intrinsic_model_id_for_hash(const uint32_t hash)
{
  if(!hash) return NULL;

  GList *models = dt_intrinsic_list_models();
  char *id = NULL;
  for(const GList *l = models; l && !id; l = g_list_next(l))
  {
    const dt_intrinsic_model_t *m = l->data;
    if(m->hash == hash) id = g_strdup(m->id);
  }
  dt_intrinsic_models_free(models);
  return id;
}

/* --- loading --- */

dt_intrinsic_env_t *dt_intrinsic_env_init(void)
{
  dt_ai_environment_t *ai = dt_ai_env_init(NULL);
  if(!ai) return NULL;

  dt_intrinsic_env_t *env = g_new0(dt_intrinsic_env_t, 1);
  env->ai_env = ai;
  return env;
}

void dt_intrinsic_env_destroy(dt_intrinsic_env_t *env)
{
  if(!env) return;
  if(env->ai_env) dt_ai_env_destroy(env->ai_env);
  g_free(env);
}

static void _read_params(const dt_ai_model_info_t *info,
                         dt_intrinsic_model_params_t *p)
{
  p->working_size = dt_ai_model_attribute_int(info, "working_size", 1024);
  if(p->working_size <= 0) p->working_size = 1024;
  p->input_normalized = dt_ai_model_attribute_bool(info, "input_normalized");
  p->split_radius_rel
    = (float)dt_ai_model_attribute_double(info, "split_radius_rel", 0.004);
  p->split_sqrt_eps
    = (float)dt_ai_model_attribute_double(info, "split_sqrt_eps", 0.3);
  p->guide_radius_factor
    = (float)dt_ai_model_attribute_double(info, "guide_radius_factor", 2.0);
  p->guide_sqrt_eps
    = (float)dt_ai_model_attribute_double(info, "guide_sqrt_eps", 0.1);
  // the bool accessor has no default and reads a missing key as FALSE;
  // json-glib hands a JSON boolean to the int accessor as 0 or 1
  p->relief = dt_ai_model_attribute_int(info, "relief", 1) != 0;
}

dt_intrinsic_ctx_t *dt_intrinsic_load(dt_intrinsic_env_t *env, const char *model_id)
{
  if(!env || !env->ai_env || !model_id) return NULL;

  const dt_ai_model_info_t *info = dt_ai_get_model_info_by_id(env->ai_env, model_id);
  if(!info)
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] model %s not found", model_id);
    return NULL;
  }
  if(g_strcmp0(info->task_type, TASK_KEY))
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] model %s has task '%s', not '%s'",
             model_id, info->task_type ? info->task_type : "", TASK_KEY);
    return NULL;
  }

  const char *arch = info->arch ? info->arch : "";
  dt_intrinsic_arch_t arch_type;
  if(strcmp(arch, "marigold") == 0)
    arch_type = DT_INTRINSIC_ARCH_MARIGOLD;
  else if(strcmp(arch, "compphoto") == 0)
    arch_type = DT_INTRINSIC_ARCH_COMPPHOTO;
  else
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] unknown arch '%s' for %s", arch, model_id);
    return NULL;
  }

  dt_intrinsic_model_params_t params;
  _read_params(info, &params);

  const double t0 = dt_get_wtime();
  dt_ai_context_t *ai_ctx
    = dt_ai_load_model(env->ai_env, model_id, NULL, DT_AI_PROVIDER_CONFIGURED);
  if(!ai_ctx)
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] failed to load %s", model_id);
    return NULL;
  }

  // every arch has one input "image" [1,3,H,W] and one output "shading"
  // [1,1,H,W], both float32 (evidence/04-summary.md)
  int64_t out_shape[4] = { 0 };
  const dt_ai_dtype_t out_type = dt_ai_get_output_type(ai_ctx, 0);
  if(dt_ai_get_input_count(ai_ctx) != 1
     || dt_ai_get_output_count(ai_ctx) != 1
     || dt_ai_get_output_shape(ai_ctx, 0, out_shape, 4) != 4
     || (out_type != DT_AI_FLOAT && out_type != DT_AI_FLOAT16))
  {
    dt_print(DT_DEBUG_AI,
             "[intrinsic] %s: expected one [1,3,H,W] input and one float"
             " [1,1,H,W] output, got %d inputs, %d outputs",
             model_id, dt_ai_get_input_count(ai_ctx), dt_ai_get_output_count(ai_ctx));
    dt_ai_unload_model(ai_ctx);
    return NULL;
  }

  dt_intrinsic_ctx_t *ctx = g_new0(dt_intrinsic_ctx_t, 1);
  ctx->ai_ctx = ai_ctx;
  ctx->arch = arch_type;
  ctx->model_id = g_strdup(model_id);
  ctx->hash = dt_intrinsic_hash(model_id);
  ctx->params = params;
  for(int d = 0; d < 4; d++)
  {
    ctx->out_shape[d] = out_shape[d];
    if(out_shape[d] <= 0) ctx->out_dynamic = TRUE;
  }
  g_mutex_init(&ctx->lock);

  dt_print(DT_DEBUG_AI,
           "[intrinsic] loaded %s: arch %s, working size %d, input normalized %s,"
           " %.1f s",
           model_id, arch, params.working_size,
           params.input_normalized ? "yes" : "no", dt_get_wtime() - t0);
  return ctx;
}

void dt_intrinsic_free(dt_intrinsic_ctx_t *ctx)
{
  if(!ctx) return;
  if(ctx->ai_ctx) dt_ai_unload_model(ctx->ai_ctx);
  g_mutex_clear(&ctx->lock);
  g_free(ctx->model_id);
  g_free(ctx);
}

const char *dt_intrinsic_model_id(const dt_intrinsic_ctx_t *ctx)
{
  return ctx ? ctx->model_id : NULL;
}

uint32_t dt_intrinsic_model_hash(const dt_intrinsic_ctx_t *ctx)
{
  return ctx ? ctx->hash : 0;
}

const dt_intrinsic_model_params_t *dt_intrinsic_params(const dt_intrinsic_ctx_t *ctx)
{
  return ctx ? &ctx->params : NULL;
}

/* --- inference --- */

static int _float_cmp(const void *a, const void *b)
{
  const float fa = *(const float *)a;
  const float fb = *(const float *)b;
  return (fa > fb) - (fa < fb);
}

// exposure normalization for a model whose graph expects it done
// (input_normalized): the scale that takes the 99th percentile of Rec.709
// luminance to 1.0, as proto/_shading_common.py normalize_exposure(), but
// with the nearest-rank percentile of epic §M0. no shipped model uses it
// (both graphs normalize internally, evidence/04-summary.md)
static gboolean _exposure_scale(const float *const rgb,
                                const size_t npix,
                                float *scale)
{
  float *const lum = dt_alloc_align_float(npix);
  if(!lum) return FALSE;

  size_t n = 0;
  float maxlum = 0.0f;
  for(size_t k = 0; k < npix; k++)
  {
    const float *const p = rgb + 3 * k;
    const float y = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    if(!dt_isfinite(y)) continue;
    lum[n++] = y;
    maxlum = fmaxf(maxlum, y);
  }

  float p99 = 0.0f;
  if(n > 0)
  {
    qsort(lum, n, sizeof(float), _float_cmp);
    p99 = lum[(size_t)(0.99 * (double)(n - 1))];
  }
  dt_free_align(lum);

  if(p99 <= 0.0f) p99 = maxlum > 0.0f ? maxlum : 1.0f;
  *scale = 1.0f / p99;
  return TRUE;
}

// bilinear sample of a sh x sw map onto dh x dw at pixel centers, epic §M2.2
// and proto/ls_ref.py _bilinear_sample(); used only if a graph returns a size
// other than its input (§M1.2), which neither shipped model does
static void _resample_bilinear(const float *const src,
                               const int sw,
                               const int sh,
                               float *const dst,
                               const int dw,
                               const int dh)
{
  for(int y = 0; y < dh; y++)
  {
    const float fy_pos
      = CLAMPS(((float)y + 0.5f) / (float)dh * (float)sh - 0.5f, 0.0f, (float)(sh - 1));
    const int y0 = (int)fy_pos;
    const int y1 = MIN(y0 + 1, sh - 1);
    const float fy = fy_pos - (float)y0;
    for(int x = 0; x < dw; x++)
    {
      const float fx_pos
        = CLAMPS(((float)x + 0.5f) / (float)dw * (float)sw - 0.5f, 0.0f, (float)(sw - 1));
      const int x0 = (int)fx_pos;
      const int x1 = MIN(x0 + 1, sw - 1);
      const float fx = fx_pos - (float)x0;
      const float top = src[(size_t)y0 * sw + x0] * (1.0f - fx) + src[(size_t)y0 * sw + x1] * fx;
      const float bot = src[(size_t)y1 * sw + x0] * (1.0f - fx) + src[(size_t)y1 * sw + x1] * fx;
      dst[(size_t)y * dw + x] = top * (1.0f - fy) + bot * fy;
    }
  }
}

// one image through a graph with the shared interface: linear sRGB CHW in,
// gray linear shading out. scale and clip_one are the input normalization
// (1 and FALSE when the graph normalizes itself)
static gboolean _run_linear_srgb(dt_intrinsic_ctx_t *ctx,
                                 const float *const rgb,
                                 const int w,
                                 const int h,
                                 const float scale,
                                 const gboolean clip_one,
                                 float *const shading)
{
  const size_t npix = (size_t)w * h;
  float *const chw = dt_alloc_align_float(3 * npix);
  if(!chw)
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] %s: out of memory for the input tensor",
             ctx->model_id);
    return FALSE;
  }

  // HWC -> CHW, negative and non-finite channels to 0
  const float hi = clip_one ? 1.0f : FLT_MAX;
  for(size_t k = 0; k < npix; k++)
    for(int c = 0; c < 3; c++)
    {
      const float v = rgb[3 * k + c] * scale;
      chw[c * npix + k] = dt_isfinite(v) ? CLAMPS(v, 0.0f, hi) : 0.0f;
    }

  int64_t in_shape[4] = { 1, 3, h, w };
  dt_ai_tensor_t input = { .data = chw, .type = DT_AI_FLOAT, .shape = in_shape, .ndim = 4 };

  // a graph with symbolic output dims gets its output allocated by the
  // backend with g_try_malloc() (src/ai/backend_onnx.c:3031); a static one
  // needs a caller buffer of its exact shape. both are released with g_free()
  int64_t out_shape[4] = { 1, 1, h, w };
  float *out = NULL;
  if(!ctx->out_dynamic)
  {
    for(int d = 0; d < 4; d++) out_shape[d] = ctx->out_shape[d];
    out = g_try_malloc_n(dt_ai_tensor_element_count(out_shape, 4), sizeof(float));
    if(!out)
    {
      dt_print(DT_DEBUG_AI, "[intrinsic] %s: out of memory for the output tensor",
               ctx->model_id);
      dt_free_align(chw);
      return FALSE;
    }
  }
  dt_ai_tensor_t output = { .data = out, .type = DT_AI_FLOAT, .shape = out_shape, .ndim = 4 };

  const int ret = dt_ai_run(ctx->ai_ctx, &input, 1, &output, 1);
  dt_free_align(chw);
  out = output.data;
  if(ret != 0 || !out)
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] %s: inference failed (%d) at %dx%d",
             ctx->model_id, ret, w, h);
    g_free(out);
    return FALSE;
  }

  if(output.ndim != 4 || out_shape[0] != 1 || out_shape[1] != 1
     || out_shape[2] <= 0 || out_shape[3] <= 0
     || out_shape[2] > G_MAXINT || out_shape[3] > G_MAXINT)
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] %s: unexpected output shape, %d dims",
             ctx->model_id, output.ndim);
    g_free(out);
    return FALSE;
  }

  const int oh = (int)out_shape[2];
  const int ow = (int)out_shape[3];
  if(oh == h && ow == w)
    memcpy(shading, out, npix * sizeof(float));
  else
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] %s: output %dx%d resampled to %dx%d",
             ctx->model_id, ow, oh, w, h);
    _resample_bilinear(out, ow, oh, shading, w, h);
  }
  g_free(out);
  return TRUE;
}

gboolean dt_intrinsic_shading(dt_intrinsic_ctx_t *ctx,
                              const float *rgb,
                              const int w,
                              const int h,
                              float *shading)
{
  if(!ctx || !rgb || !shading || w <= 0 || h <= 0) return FALSE;

  const double t0 = dt_get_wtime();
  const size_t npix = (size_t)w * h;
  gboolean ok = FALSE;

  g_mutex_lock(&ctx->lock);

  float scale = 1.0f;
  if(ctx->params.input_normalized && !_exposure_scale(rgb, npix, &scale))
  {
    dt_print(DT_DEBUG_AI, "[intrinsic] %s: exposure normalization failed"
             " (out of memory)", ctx->model_id);
    goto done;
  }

  switch(ctx->arch)
  {
    case DT_INTRINSIC_ARCH_MARIGOLD:
      // tasks/light-and-surface/epic-01/evidence/04-onnx-marigold.md, "What C
      // does": the graph holds the whole pipeline (exposure normalization,
      // gamma, VAE, 4 DDIM steps, decode, resize, gray). C only supplies
      // linear sRGB clipped at 0 in CHW layout; the output is at input size
      ok = _run_linear_srgb(ctx, rgb, w, h, scale, ctx->params.input_normalized, shading);
      break;
    case DT_INTRINSIC_ARCH_COMPPHOTO:
      // tasks/light-and-surface/epic-01/evidence/04-onnx-compphoto.md, "What C
      // does": the graph holds exposure normalization, the resolution search,
      // every resize, all five networks and the resample back to the input
      // size. C does the same layout-only step as for marigold
      ok = _run_linear_srgb(ctx, rgb, w, h, scale, ctx->params.input_normalized, shading);
      break;
  }

  if(ok)
    for(size_t k = 0; k < npix; k++)
      if(!(shading[k] > 0.0f) || !dt_isfinite(shading[k])) shading[k] = SHADING_FLOOR;

done:
  g_mutex_unlock(&ctx->lock);

  if(ok)
    dt_print(DT_DEBUG_AI, "[intrinsic] %s: %dx%d in %.0f ms",
             ctx->model_id, w, h, 1000.0 * (dt_get_wtime() - t0));
  return ok;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
