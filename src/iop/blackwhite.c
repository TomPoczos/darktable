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
#include "common/imagebuf.h"
#include "common/math.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"


DT_MODULE_INTROSPECTION(1, dt_iop_blackwhite_params_t)

typedef struct dt_iop_blackwhite_params_t
{
  gboolean filter; // $DEFAULT: FALSE $DESCRIPTION: "color filter"
  float hue;       // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 0.0 $DESCRIPTION: "hue"
  float chroma;    // $MIN: 0.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "chroma"
} dt_iop_blackwhite_params_t;

typedef struct dt_iop_blackwhite_gui_data_t
{
  GtkWidget *filter;
  GtkWidget *swatch;
  GtkWidget *hue;
  GtkWidget *chroma;
} dt_iop_blackwhite_gui_data_t;

typedef struct dt_iop_blackwhite_data_t
{
  gboolean filter;
  dt_aligned_pixel_t grey; // normalized R/G/B mix weights, 4th component always 0
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
  dt_iop_blackwhite_params_t p = { .filter = FALSE, .hue = 0.f, .chroma = 0.f };
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

  dt_aligned_pixel_t grey = { 1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 0.f };
  if(p->filter)
  {
    const float hue = deg2radf(p->hue);
    const float chroma = p->chroma;
    for(int c = 0; c < 3; c++)
      grey[c] = 1.f / 3.f + chroma * cosf(hue - c * (2.f * M_PI_F / 3.f));
  }

  float norm = grey[0] + grey[1] + grey[2];
  if(norm == 0.f) norm = 1.f;

  for(int c = 0; c < 3; c++) d->grey[c] = grey[c] / norm;
  d->grey[3] = 0.f;
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
  const size_t npixels = (size_t)roi_out->width * roi_out->height;
  const float *const restrict in = (const float *const)ivoid;
  float *const restrict out = (float *const)ovoid;
  const dt_aligned_pixel_t grey = { d->grey[0], d->grey[1], d->grey[2], d->grey[3] };

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float *const px = in + 4 * k;
    float *const o = out + 4 * k;
    const float grey_mix = fmaxf(px[0] * grey[0] + px[1] * grey[1] + px[2] * grey[2], 0.f);
    o[0] = o[1] = o[2] = grey_mix;
    o[3] = px[3];
  }
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

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const dt_aligned_pixel_t grey = { d->grey[0], d->grey[1], d->grey[2], d->grey[3] };

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

// same shape as commit_params' grey[] computation, but clamped to a displayable
// [0, 1] range instead of normalized to sum=1 -- so the swatch reads as neutral
// grey at chroma=0 and grows more saturated as chroma increases, the way an
// actual colored filter would look.
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
  gtk_widget_set_tooltip_text(g->hue, _("hue of the virtual color filter"));
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
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
