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

// src/common/ai/intrinsic.h
#pragma once
#include <glib.h>
#include <stdint.h>

typedef struct dt_intrinsic_env_t dt_intrinsic_env_t;
typedef struct dt_intrinsic_ctx_t dt_intrinsic_ctx_t;

// per-model tuning, from config.json "attributes" (epic §M0); fallbacks when absent
typedef struct dt_intrinsic_model_params_t
{
  int working_size;          // "working_size", default 1024
  gboolean input_normalized; // "input_normalized", default FALSE
  float split_radius_rel;    // "split_radius_rel", default 0.004
  float split_sqrt_eps;      // "split_sqrt_eps", default 0.3 (EV)
  float guide_radius_factor; // "guide_radius_factor", default 2.0
  float guide_sqrt_eps;      // "guide_sqrt_eps", default 0.1 (EV, log guide)
  gboolean relief;           // "relief", default TRUE
} dt_intrinsic_model_params_t;

typedef struct dt_intrinsic_model_t
{
  char *id;       // e.g. "intrinsic-ordinal"
  char *name;     // display name from config.json, for the combobox
  uint32_t hash;  // dt_intrinsic_hash(id)
} dt_intrinsic_model_t;

// 32-bit hash of a model id, never 0 (0 means "none"); same algorithm as
// spektrafilm's _name_hash()
uint32_t dt_intrinsic_hash(const char *model_id);

// TRUE if AI is enabled in preferences and at least one "intrinsic" model is
// discoverable; cheap enough for reload_defaults() and gui_update()
gboolean dt_intrinsic_available(void);

// all discoverable models of task "intrinsic", sorted by name; a GList of
// dt_intrinsic_model_t*; free with dt_intrinsic_models_free()
GList *dt_intrinsic_list_models(void);
void dt_intrinsic_models_free(GList *models);

// the model id active for the task in preferences (newly allocated) or NULL
char *dt_intrinsic_active_model_id(void);

// the id (newly allocated) of the discoverable model with this hash, or NULL
char *dt_intrinsic_model_id_for_hash(const uint32_t hash);

dt_intrinsic_env_t *dt_intrinsic_env_init(void);
void dt_intrinsic_env_destroy(dt_intrinsic_env_t *env);

// loads one model by id; NULL on failure (unknown arch included)
dt_intrinsic_ctx_t *dt_intrinsic_load(dt_intrinsic_env_t *env, const char *model_id);
void dt_intrinsic_free(dt_intrinsic_ctx_t *ctx);

const char *dt_intrinsic_model_id(const dt_intrinsic_ctx_t *ctx);
uint32_t dt_intrinsic_model_hash(const dt_intrinsic_ctx_t *ctx);
const dt_intrinsic_model_params_t *dt_intrinsic_params(const dt_intrinsic_ctx_t *ctx);

// rgb: linear sRGB, 3 floats per pixel, row-major (HWC), w x h, values >= 0.
// shading: caller-allocated w * h floats; receives gray linear shading > 0.
// thread-safe: concurrent calls on one ctx are serialized internally.
gboolean dt_intrinsic_shading(dt_intrinsic_ctx_t *ctx,
                              const float *rgb,
                              const int w,
                              const int h,
                              float *shading);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
