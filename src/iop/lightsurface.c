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
#endif

#include <gtk/gtk.h>
#include <string.h>

DT_MODULE_INTROSPECTION(1, dt_iop_lightsurface_params_t)

// light curve nodes sit at x_i = -7 + i EV; surface hue nodes at 45 degree steps
#define LS_LIGHT_NODES 9
#define LS_HUE_NODES 8

// minimum output rise between adjacent light nodes, in EV
#define LS_NODE_MIN_STEP 0.1f

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

typedef struct dt_iop_lightsurface_gui_data_t
{
  GtkWidget *form, *relief, *fill;
  GtkWidget *light_node[LS_LIGHT_NODES];
  GtkWidget *surface_contrast;
  GtkWidget *surface_hue[LS_HUE_NODES];
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

  dt_iop_image_copy_by_size((float *)ovoid, (const float *)ivoid,
                            roi_out->width, roi_out->height, 4);
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
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
