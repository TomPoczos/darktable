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

/* Film emulation tone mapper -- a live, native port of the PoLUT project's
 * physically-grounded film-emulation color science (real digitized spectral
 * sensitivities and H&D characteristic curves, per-pixel spectral
 * reconstruction via Jakob & Hanika 2019, and a real print-paper cascade)
 * into darktable's own scene->display tone-mapping stage, as a sibling of
 * filmicrgb/agx/sigmoid rather than a static 3D LUT. See that project's
 * generate_film_looks.py and README.md for the full darkroom/color-science
 * rationale and citation trail -- not repeated here.
 *
 * Every film/paper/filter's numeric data lives in external/filmemulation_data.c
 * (baked static const tables, exported from PoLUT -- see that file's own
 * header comment). This file only implements the runtime algorithm and GUI.
 */

#include "bauhaus/bauhaus.h"
#include "common/chromatic_adaptation.h"
#include "common/colorspaces.h"
#include "common/gamut_mapping.h"
#include "common/imagebuf.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/matrices.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <math.h>
#include <stdlib.h>

#include "external/filmemulation_data.c"

DT_MODULE_INTROSPECTION(1, dt_iop_filmemulation_params_t)

// ---------------------------------------------------------------------------
// Params enums
// ---------------------------------------------------------------------------
typedef enum dt_iop_filmemulation_type_t
{
  DT_FE_TYPE_BW = 0,    // $DESCRIPTION: "black & white"
  DT_FE_TYPE_COLOR = 1, // $DESCRIPTION: "color"
} dt_iop_filmemulation_type_t;

typedef enum dt_iop_filmemulation_bw_film_t
{
  DT_FE_BW_TRIX400 = 0, // $DESCRIPTION: "Kodak Tri-X 400"
} dt_iop_filmemulation_bw_film_t;

typedef enum dt_iop_filmemulation_bw_filter_t
{
  DT_FE_FILTER_NONE = 0,     // $DESCRIPTION: "none"
  DT_FE_FILTER_YELLOW8 = 1,  // $DESCRIPTION: "yellow (Wratten 8)"
  DT_FE_FILTER_ORANGE21 = 2, // $DESCRIPTION: "orange (Wratten 21)"
  DT_FE_FILTER_RED25 = 3,    // $DESCRIPTION: "red (Wratten 25)"
  DT_FE_FILTER_GREEN58 = 4,  // $DESCRIPTION: "green (Wratten 58)"
  DT_FE_FILTER_BLUE47 = 5,   // $DESCRIPTION: "blue (Wratten 47)"
} dt_iop_filmemulation_bw_filter_t;

typedef enum dt_iop_filmemulation_process_t
{
  DT_FE_PROCESS_REVERSAL = 0, // $DESCRIPTION: "reversal (slide)"
  DT_FE_PROCESS_NEGATIVE = 1, // $DESCRIPTION: "negative"
} dt_iop_filmemulation_process_t;

typedef enum dt_iop_filmemulation_route_t
{
  DT_FE_ROUTE_DIRECT = 0,        // $DESCRIPTION: "straight to paper"
  DT_FE_ROUTE_INTERNEGATIVE = 1, // $DESCRIPTION: "internegative"
} dt_iop_filmemulation_route_t;

typedef enum dt_iop_filmemulation_reversal_film_t
{
  DT_FE_REV_VELVIA50 = 0,        // $DESCRIPTION: "Velvia 50"
  DT_FE_REV_KODACHROME64 = 1,    // $DESCRIPTION: "Kodachrome 64"
  DT_FE_REV_PROVIA100F = 2,      // $DESCRIPTION: "Fuji Provia 100F"
  DT_FE_REV_EKTACHROME100D = 3,  // $DESCRIPTION: "Kodak Ektachrome 100D"
} dt_iop_filmemulation_reversal_film_t;

typedef enum dt_iop_filmemulation_negative_film_t
{
  DT_FE_NEG_PORTRA400 = 0,       // $DESCRIPTION: "Kodak Portra 400"
  DT_FE_NEG_EKTAR100 = 1,        // $DESCRIPTION: "Kodak Ektar 100"
  DT_FE_NEG_GOLD200 = 2,         // $DESCRIPTION: "Kodak Gold 200"
  DT_FE_NEG_ULTRAMAX400 = 3,     // $DESCRIPTION: "Kodak Ultramax 400"
  DT_FE_NEG_SUPERIA_REALA = 4,   // $DESCRIPTION: "Fuji Superia Reala"
  DT_FE_NEG_SUPERIA_XTRA400 = 5, // $DESCRIPTION: "Fuji Superia X-tra 400"
} dt_iop_filmemulation_negative_film_t;

typedef enum dt_iop_filmemulation_direct_paper_t
{
  DT_FE_DP_RADIANCE_III = 0, // $DESCRIPTION: "Kodak Ektachrome Radiance III"
  DT_FE_DP_ILFOCHROME_M = 1, // $DESCRIPTION: "Ilfochrome Micrographic M"
  DT_FE_DP_ILFOCHROME_P = 2, // $DESCRIPTION: "Ilfochrome Micrographic P"
} dt_iop_filmemulation_direct_paper_t;

// The 5-rung paper ladder used by negative films and the reversal
// internegative route -- real material names, NOT the ExtraSoft/Soft/.../
// ExtraPunchy look-names PoLUT itself uses internally: those look-names
// describe contrast/punchiness, which is now the live print gamma slider's job.
typedef enum dt_iop_filmemulation_ladder_paper_t
{
  DT_FE_LADDER_SUPER_TYPE_C = 0,  // $DESCRIPTION: "Fuji Crystal Archive Super Type C"
  DT_FE_LADDER_PRO_PDII = 1,      // $DESCRIPTION: "Fuji Crystal Archive Pro PDII"
  DT_FE_LADDER_PORTRA_ENDURA = 2, // $DESCRIPTION: "Kodak Portra Endura"
  DT_FE_LADDER_DPII = 3,          // $DESCRIPTION: "Fuji Crystal Archive DPII"
  DT_FE_LADDER_SUPRA_ENDURA = 4,  // $DESCRIPTION: "Kodak Supra Endura"
} dt_iop_filmemulation_ladder_paper_t;

typedef struct dt_iop_filmemulation_params_t
{
  dt_iop_filmemulation_type_t type; // $DEFAULT: DT_FE_TYPE_COLOR $DESCRIPTION: "type"

  dt_iop_filmemulation_bw_film_t bw_film;     // $DEFAULT: DT_FE_BW_TRIX400 $DESCRIPTION: "film"
  dt_iop_filmemulation_bw_filter_t bw_filter; // $DEFAULT: DT_FE_FILTER_NONE $DESCRIPTION: "filter"

  dt_iop_filmemulation_process_t process; // $DEFAULT: DT_FE_PROCESS_NEGATIVE $DESCRIPTION: "process"

  dt_iop_filmemulation_reversal_film_t reversal_film; // $DEFAULT: DT_FE_REV_VELVIA50 $DESCRIPTION: "film"
  dt_iop_filmemulation_route_t route;                 // $DEFAULT: DT_FE_ROUTE_INTERNEGATIVE $DESCRIPTION: "development"
  dt_iop_filmemulation_direct_paper_t direct_paper;    // $DEFAULT: DT_FE_DP_RADIANCE_III $DESCRIPTION: "paper"
  dt_iop_filmemulation_ladder_paper_t internegative_paper; // $DEFAULT: DT_FE_LADDER_PORTRA_ENDURA $DESCRIPTION: "paper"

  dt_iop_filmemulation_negative_film_t negative_film; // $DEFAULT: DT_FE_NEG_PORTRA400 $DESCRIPTION: "film"
  dt_iop_filmemulation_ladder_paper_t negative_paper;  // $DEFAULT: DT_FE_LADDER_PORTRA_ENDURA $DESCRIPTION: "paper"

  float print_gamma; // $MIN: 1.0 $MAX: 1.6 $DEFAULT: 1.25 $DESCRIPTION: "print gamma"
  gboolean use_hk; // $DEFAULT: FALSE $DESCRIPTION: "perceptual color correction (HK)"
} dt_iop_filmemulation_params_t;

typedef struct dt_iop_filmemulation_gui_data_t
{
  GtkWidget *type;
  GtkWidget *box_bw, *bw_film, *bw_filter;
  GtkWidget *box_color, *process;
  GtkWidget *box_negative, *negative_film, *negative_paper;
  GtkWidget *box_reversal, *reversal_film, *route;
  GtkWidget *box_direct, *direct_paper;
  GtkWidget *box_internegative, *internegative_paper;
  GtkWidget *print_gamma, *use_hk;
} dt_iop_filmemulation_gui_data_t;

// ---------------------------------------------------------------------------
// Pipeline (piece->data) representation of a resolved cascade -- one entry
// per color layer (1 for B&W, 3 for color). All the "search"/calibration
// work (_fe_find_anchor, gamma-correction factor derivation) happens once
// here, in commit_params(); the per-pixel hot path (_fe_xfer()) is just a
// couple of small interpolations/closed-form evaluations.
// ---------------------------------------------------------------------------
#define FE_MAX_STAGES 3

typedef struct dt_iop_filmemulation_stage_data_t
{
  const dt_film_curve_point_t *pts;
  int n;
  gboolean increasing;
} dt_iop_filmemulation_stage_data_t;

typedef struct dt_iop_filmemulation_layer_data_t
{
  float weight[FE_N_WAVELENGTHS]; // normalized sens*D65[*filter] exposure weights

  int n_stages;      // 2 or 3
  int corrected_stage; // always n_stages - 2
  dt_iop_filmemulation_stage_data_t stage[FE_MAX_STAGES]; // stage[corrected_stage] unused
  dt_film_splitgauss_t fit; // corrected stage's original (uncorrected) fitted model
  float k_lo, k_hi;         // live gamma-driven correction factors
  float pls[FE_MAX_STAGES - 1];
  float na0;
  float x0_sentinel;
  float fdm;
} dt_iop_filmemulation_layer_data_t;

typedef struct dt_iop_filmemulation_data_t
{
  int n_layers; // 1 (B&W) or 3 (color)
  dt_iop_filmemulation_layer_data_t layer[3];
  gboolean use_hk;
} dt_iop_filmemulation_data_t;

// ---------------------------------------------------------------------------
// Module metadata
// ---------------------------------------------------------------------------
const char *name()
{
  return _("film emulation");
}

const char *aliases()
{
  return _("tone mapping|view transform|display transform|film|analog");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
                                _("apply a physically-grounded film and print-paper look\n"
                                  "using real digitized spectral sensitivities and\n"
                                  "characteristic curves instead of a synthetic tone curve"),
                                _("corrective and creative"), _("linear, RGB, scene-referred"),
                                _("non-linear, RGB"), _("linear, RGB, display-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_TONE | IOP_GROUP_TECHNICAL;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

// ---------------------------------------------------------------------------
// Math helpers -- direct ports of generate_film_looks.py's own functions of
// the same name (see that file for the physical justification of each).
// ---------------------------------------------------------------------------

// Linear interpolation, clamped to endpoint values outside range -- matches
// Python's _il(). Used for H&D curves and filter transmission curves.
static inline float _fe_lerp_clamped(const dt_film_curve_point_t *pts, const int n, const float x)
{
  if(x <= pts[0].x) return pts[0].y;
  if(x >= pts[n - 1].x) return pts[n - 1].y;
  int lo = 0, hi = n - 1;
  while(hi - lo > 1)
  {
    const int mid = (lo + hi) / 2;
    if(pts[mid].x <= x) lo = mid; else hi = mid;
  }
  const float t = (x - pts[lo].x) / (pts[hi].x - pts[lo].x);
  return pts[lo].y * (1.f - t) + pts[hi].y * t;
}

// Linear interpolation of a LOG10 curve, returning 10**value, zero outside
// range -- matches Python's _il10(). Used for spectral sensitivity curves.
static inline float _fe_lerp_log10_zero_outside(const dt_film_curve_point_t *pts, const int n, const float wl)
{
  if(wl < pts[0].x || wl > pts[n - 1].x) return 0.f;
  int lo = 0, hi = n - 1;
  while(hi - lo > 1)
  {
    const int mid = (lo + hi) / 2;
    if(pts[mid].x <= wl) lo = mid; else hi = mid;
  }
  const float t = (wl - pts[lo].x) / (pts[hi].x - pts[lo].x);
  const float logval = pts[lo].y * (1.f - t) + pts[hi].y * t;
  return powf(10.f, logval);
}

// Find x where a digitized curve crosses target density td -- matches
// Python's _find_anchor(). Setup-time only (called from commit_params).
static float _fe_find_anchor(const dt_film_curve_point_t *pts, const int n, const float td,
                             const gboolean increasing, const int start)
{
  if(increasing)
  {
    if(td <= pts[start].y) return pts[start].x;
    if(td >= pts[n - 1].y) return pts[n - 1].x;
    for(int i = start; i < n - 1; i++)
    {
      if(pts[i].y == pts[i + 1].y) { if(td == pts[i].y) return pts[i].x; continue; }
      if(pts[i].y <= td && td <= pts[i + 1].y)
      {
        const float t = (td - pts[i].y) / (pts[i + 1].y - pts[i].y);
        return pts[i].x * (1.f - t) + pts[i + 1].x * t;
      }
    }
  }
  else
  {
    if(td >= pts[start].y) return pts[start].x;
    if(td <= pts[n - 1].y) return pts[n - 1].x;
    for(int i = start; i < n - 1; i++)
    {
      if(pts[i].y == pts[i + 1].y) { if(td == pts[i].y) return pts[i].x; continue; }
      if(pts[i].y >= td && td >= pts[i + 1].y)
      {
        const float t = (td - pts[i].y) / (pts[i + 1].y - pts[i].y);
        return pts[i].x * (1.f - t) + pts[i + 1].x * t;
      }
    }
  }
  // Baked data is pre-validated monotonic in the region any caller here
  // scans; fall back to the far endpoint rather than crash if that ever
  // changes.
  return pts[n - 1].x;
}

// Matches Python's _detect_lead_noise_start(): how many leading samples of
// a digitized curve are Dmin/Dmax-plateau digitization noise, not the real
// monotonic climb/decline.
static int _fe_detect_lead_noise_start(const dt_film_curve_point_t *pts, const int n, const gboolean increasing)
{
  const int max_lead = 15;
  const int lim = (max_lead < n - 1) ? max_lead : n - 1;
  int last_bad = -1;
  for(int i = 0; i < lim; i++)
  {
    const gboolean bad = increasing ? (pts[i].y > pts[i + 1].y) : (pts[i].y < pts[i + 1].y);
    if(bad) last_bad = i;
  }
  return last_bad + 1;
}

// |dD/dx| at x via a centered secant on a real digitized curve -- matches
// Python's _digitized_local_gamma() (h=0.15, ~1/2 stop in log10(E) units).
static inline float _fe_digitized_local_gamma(const dt_film_curve_point_t *pts, const int n, const float x)
{
  const float h = 0.15f;
  return fabsf(_fe_lerp_clamped(pts, n, x + h) - _fe_lerp_clamped(pts, n, x - h)) / (2.f * h);
}

static inline float _fe_norm_cdf(const float z)
{
  return 0.5f * (1.f + erff(z / sqrtf(2.f)));
}

static inline float _fe_norm_pdf(const float z)
{
  return expf(-z * z / 2.f) / sqrtf(2.f * (float)M_PI);
}

// Inverse standard normal CDF via bisection -- matches Python's _norm_ppf().
static float _fe_norm_ppf(const float p)
{
  float lo = -8.f, hi = 8.f;
  for(int i = 0; i < 60; i++)
  {
    const float mid = 0.5f * (lo + hi);
    if(_fe_norm_cdf(mid) < p) lo = mid; else hi = mid;
  }
  return 0.5f * (lo + hi);
}

static inline float _fe_split_gauss_density(const dt_film_splitgauss_t *f, const float x)
{
  const float sigma = (x < f->x0) ? f->sigma_lo : f->sigma_hi;
  return f->d_lo + (f->d_hi - f->d_lo) * _fe_norm_cdf((x - f->x0) / sigma);
}

// ---------------------------------------------------------------------------
// Cascade setup (commit_params time) -- direct port of generate_film_looks.py's
// build_print_cascade()/gamma_correct_curve(), specialized: the stage
// immediately upstream of the final paper (index n_stages-2, always) is
// evaluated analytically from its fitted split-Gaussian model, rescaled live
// by the print gamma slider; every other stage is a plain real-curve
// interpolation. See the plan/PR description for why this is a uniform rule
// across all four routes (B&W, reversal direct-print, negative, reversal
// internegative) even though generate_film_looks.py itself only gamma-
// corrects two of those four.
// ---------------------------------------------------------------------------
static void _fe_setup_layer(dt_iop_filmemulation_layer_data_t *L,
                            const dt_film_curve_t *sens_curve,
                            const dt_film_curve_t *filter_curve, // NULL if none
                            const int n_stages,
                            const dt_film_curve_t *stage_curve[FE_MAX_STAGES], // corrected index ignored
                            const gboolean stage_increasing[FE_MAX_STAGES],
                            const float stage_ref_d[FE_MAX_STAGES], // ignored for final stage
                            const dt_film_splitgauss_t *corrected_fit,
                            const float gamma_target)
{
  // 1. Exposure weight vector: sens(wl)*D65(wl)[*filter(wl)/100], normalized
  // to sum 1 so that an achromatic grey pixel's exactly-flat reconstructed
  // spectrum yields exposure == the grey reflectance value itself.
  float wsum = 0.f;
  for(int i = 0; i < FE_N_WAVELENGTHS; i++)
  {
    const float wl = fe_wavelengths[i];
    float s = _fe_lerp_log10_zero_outside(sens_curve->pts, sens_curve->n, wl);
    if(filter_curve) s *= _fe_lerp_clamped(filter_curve->pts, filter_curve->n, wl) / 100.f;
    const float wgt = s * fe_d65[i];
    L->weight[i] = wgt;
    wsum += wgt;
  }
  if(wsum > 0.f)
    for(int i = 0; i < FE_N_WAVELENGTHS; i++) L->weight[i] /= wsum;

  L->n_stages = n_stages;
  const int corrected = n_stages - 2;
  L->corrected_stage = corrected;

  int start[FE_MAX_STAGES] = { 0, 0, 0 };
  for(int i = 0; i < n_stages; i++)
  {
    if(i == corrected) continue;
    L->stage[i].pts = stage_curve[i]->pts;
    L->stage[i].n = stage_curve[i]->n;
    L->stage[i].increasing = stage_increasing[i];
    start[i] = _fe_detect_lead_noise_start(stage_curve[i]->pts, stage_curve[i]->n, stage_increasing[i]);
  }

  // Final stage: grey target density, its own real grey-crossing exposure
  // (lhg), and the downstream local gamma the correction targets against.
  const dt_film_curve_t *final_curve = stage_curve[n_stages - 1];
  float fdm = final_curve->pts[0].y;
  for(int i = 1; i < final_curve->n; i++)
    if(final_curve->pts[i].y < fdm) fdm = final_curve->pts[i].y;
  L->fdm = fdm;
  const float grey_target_d = fdm - log10f(0.18f);
  const int fstart = _fe_detect_lead_noise_start(final_curve->pts, final_curve->n, stage_increasing[n_stages - 1]);
  const float lhg = _fe_find_anchor(final_curve->pts, final_curve->n, grey_target_d,
                                    stage_increasing[n_stages - 1], fstart);
  const float downstream_gamma = _fe_digitized_local_gamma(final_curve->pts, final_curve->n, lhg);

  // Anchor exposure (na) for every non-final stage.
  float na[FE_MAX_STAGES] = { 0.f, 0.f, 0.f };
  for(int i = 0; i < n_stages - 1; i++)
  {
    if(i == corrected) continue;
    na[i] = _fe_find_anchor(L->stage[i].pts, L->stage[i].n, stage_ref_d[i], L->stage[i].increasing, start[i]);
  }

  // Corrected stage: derive k_lo/k_hi (Jones-rule criterion evaluated at the
  // real grey operating point z_ref, not the toe/shoulder junction -- Ticket
  // 19's fix) and this stage's own anchor exposure, in closed form.
  L->fit = *corrected_fit;
  const float ref_d = stage_ref_d[corrected];
  const float z_ref = _fe_norm_ppf((ref_d - corrected_fit->d_lo) / (corrected_fit->d_hi - corrected_fit->d_lo));
  const float peak = fabsf(corrected_fit->d_hi - corrected_fit->d_lo) * _fe_norm_pdf(z_ref);
  L->k_lo = gamma_target / (downstream_gamma * (peak / corrected_fit->sigma_lo));
  L->k_hi = gamma_target / (downstream_gamma * (peak / corrected_fit->sigma_hi));
  {
    const float sigma = (z_ref < 0.f) ? corrected_fit->sigma_lo : corrected_fit->sigma_hi;
    const float k = (z_ref < 0.f) ? L->k_lo : L->k_hi;
    na[corrected] = corrected_fit->x0 + z_ref * sigma / k;
  }

  // Stage-to-stage transition constants ("printer light").
  for(int i = 0; i < n_stages - 1; i++)
  {
    if(i == n_stages - 2) L->pls[i] = lhg + stage_ref_d[i];
    else L->pls[i] = na[i + 1] + stage_ref_d[i];
  }

  L->na0 = na[0];
  if(corrected == 0)
    L->x0_sentinel = corrected_fit->x0 - 8.f * corrected_fit->sigma_lo / L->k_lo - 10.f;
  else
    L->x0_sentinel = L->stage[0].pts[0].x - 10.f;
}

// Evaluate one cascade stage's density at exposure/log-position x.
static inline float _fe_eval_stage(const dt_iop_filmemulation_layer_data_t *L, const int stage_idx, const float x)
{
  if(stage_idx == L->corrected_stage)
  {
    const float k = (x < L->fit.x0) ? L->k_lo : L->k_hi;
    const float x_orig = L->fit.x0 + (x - L->fit.x0) * k;
    return _fe_split_gauss_density(&L->fit, x_orig);
  }
  return _fe_lerp_clamped(L->stage[stage_idx].pts, L->stage[stage_idx].n, x);
}

// Full cascade transfer function E (linear exposure) -> reflectance --
// matches Python's build_print_cascade()'s inner xfer() closure exactly.
// Deliberately NOT clamped to [0,1] here: a highlight can legitimately
// reconstruct to a reflectance above 1 (or, per-channel, produce an
// out-of-gamut chromaticity once the three layers are recombined) and
// chopping each channel independently right here is exactly the naive
// per-channel clamp _fe_gamut_map() below replaces with a hue-preserving
// gamut compression once all three layers are assembled.
static inline float _fe_xfer(const dt_iop_filmemulation_layer_data_t *L, const float E)
{
  const float lh = (E <= 1e-9f) ? L->x0_sentinel : (L->na0 + log10f(E / fe_grey));
  float D = _fe_eval_stage(L, 0, lh);
  for(int i = 1; i < L->n_stages; i++)
    D = _fe_eval_stage(L, i, L->pls[i - 1] - D);
  const float v = powf(10.f, -(D - L->fdm));
  return fmaxf(v, 0.f);
}

// ---------------------------------------------------------------------------
// Per-pixel spectral reconstruction (Jakob & Hanika 2019) -- direct port of
// generate_film_looks.py's _rgb_to_sigmoid_coeffs()/_sigmoid_spectrum().
// ---------------------------------------------------------------------------
static void _fe_rgb_to_sigmoid_coeffs(float r, float g, float b, float coeffs[3])
{
  r = fminf(fmaxf(r, 0.f), 1.f);
  g = fminf(fmaxf(g, 0.f), 1.f);
  b = fminf(fmaxf(b, 0.f), 1.f);

  if(r == g && g == b)
  {
    const float v = r;
    if(v <= 0.f) { coeffs[0] = 0.f; coeffs[1] = 0.f; coeffs[2] = -1e6f; return; }
    if(v >= 1.f) { coeffs[0] = 0.f; coeffs[1] = 0.f; coeffs[2] = 1e6f; return; }
    coeffs[0] = 0.f; coeffs[1] = 0.f; coeffs[2] = (v - 0.5f) / sqrtf(v * (1.f - v));
    return;
  }

  const float rgb[3] = { r, g, b };
  int l = 0;
  float vmax = rgb[0];
  if(rgb[1] > vmax) { vmax = rgb[1]; l = 1; }
  if(rgb[2] > vmax) { vmax = rgb[2]; l = 2; }
  if(vmax <= 1e-10f) { coeffs[0] = 0.f; coeffs[1] = 0.f; coeffs[2] = -8192.f; return; }

  const float chroma[3] = { rgb[0] / vmax, rgb[1] / vmax, rgb[2] / vmax };
  const float y1 = chroma[(l + 1) % 3];
  const float y2 = chroma[(l + 2) % 3];

  const int size = FE_SPECTRAL_TABLE_SIZE;
  const float xg = fminf(fmaxf(y2 * (size - 1), 0.f), (float)(size - 1));
  const float yg = fminf(fmaxf(y1 * (size - 1), 0.f), (float)(size - 1));
  int x0 = (int)xg; if(x0 > size - 2) x0 = size - 2;
  int y0 = (int)yg; if(y0 > size - 2) y0 = size - 2;
  const int x1 = x0 + 1, y1i = y0 + 1;
  const float tx = xg - x0, ty = yg - y0;

  int lo = 0, hi = size - 1;
  while(hi - lo > 1)
  {
    const int mid = (lo + hi) / 2;
    if(fe_spectral_lightness_scale[mid] <= vmax) lo = mid; else hi = mid;
  }
  int zi = lo; if(zi > size - 2) zi = size - 2;
  const float denom = fe_spectral_lightness_scale[zi + 1] - fe_spectral_lightness_scale[zi];
  const float tz = (denom > 0.f) ? (vmax - fe_spectral_lightness_scale[zi]) / denom : 0.f;

  for(int k = 0; k < 3; k++)
  {
    const float c00 = fe_spectral_coeffs[l][zi][y0][x0][k];
    const float c01 = fe_spectral_coeffs[l][zi][y0][x1][k];
    const float c10 = fe_spectral_coeffs[l][zi][y1i][x0][k];
    const float c11 = fe_spectral_coeffs[l][zi][y1i][x1][k];
    const float d00 = fe_spectral_coeffs[l][zi + 1][y0][x0][k];
    const float d01 = fe_spectral_coeffs[l][zi + 1][y0][x1][k];
    const float d10 = fe_spectral_coeffs[l][zi + 1][y1i][x0][k];
    const float d11 = fe_spectral_coeffs[l][zi + 1][y1i][x1][k];
    const float lo_z = (c00 * (1.f - tx) + c01 * tx) * (1.f - ty) + (c10 * (1.f - tx) + c11 * tx) * ty;
    const float hi_z = (d00 * (1.f - tx) + d01 * tx) * (1.f - ty) + (d10 * (1.f - tx) + d11 * tx) * ty;
    coeffs[k] = lo_z * (1.f - tz) + hi_z * tz;
  }
}

static inline void _fe_sigmoid_spectrum(const float coeffs[3], float spectrum[FE_N_WAVELENGTHS])
{
  const float c0 = coeffs[0], c1 = coeffs[1], c2 = coeffs[2];
  for(int i = 0; i < FE_N_WAVELENGTHS; i++)
  {
    const float wl = fe_wavelengths[i];
    const float u = c0 * wl * wl + c1 * wl + c2;
    spectrum[i] = 0.5f + u / (2.f * sqrtf(1.f + u * u));
  }
}

// ---------------------------------------------------------------------------
// Helmholtz-Kohlrausch exposure multiplier (Fairchild & Pirrotta 1991),
// applied to LINEAR Adobe-RGB-primaries input -- direct port of
// generate_film_looks.py's hk_mul(), including its HK_MAX_MUL ceiling.
// ---------------------------------------------------------------------------
static inline float _fe_lab_f(const float t)
{
  const float d3 = powf(6.f / 29.f, 3.f);
  return (t > d3) ? cbrtf(t) : (t / (3.f * (6.f / 29.f) * (6.f / 29.f)) + 4.f / 29.f);
}

static inline float _fe_lab_fi(const float t)
{
  return (t > 6.f / 29.f) ? (t * t * t) : (3.f * (6.f / 29.f) * (6.f / 29.f) * (t - 4.f / 29.f));
}

static float _fe_hk_mul(const float R, const float G, const float B)
{
  if(R < 1e-6f && G < 1e-6f && B < 1e-6f) return 1.f;

  const float X = fe_adobergb_rgb2xyz[0][0] * R + fe_adobergb_rgb2xyz[0][1] * G + fe_adobergb_rgb2xyz[0][2] * B;
  const float Y = fe_adobergb_rgb2xyz[1][0] * R + fe_adobergb_rgb2xyz[1][1] * G + fe_adobergb_rgb2xyz[1][2] * B;
  const float Z = fe_adobergb_rgb2xyz[2][0] * R + fe_adobergb_rgb2xyz[2][1] * G + fe_adobergb_rgb2xyz[2][2] * B;
  if(Y < 1e-6f) return 1.f;

  const float Xn = 0.95047f, Yn = 1.0f, Zn = 1.08883f;
  const float fy = _fe_lab_f(Y / Yn);
  const float L = 116.f * fy - 16.f;
  const float a = 500.f * (_fe_lab_f(X / Xn) - fy);
  const float b = 200.f * (fy - _fe_lab_f(Z / Zn));
  const float C = sqrtf(a * a + b * b);
  float h = atan2f(b, a) * 180.f / (float)M_PI;
  h = fmodf(fmodf(h, 360.f) + 360.f, 360.f);
  if(L <= 0.01f || C < 0.5f) return 1.f;

  const float dL = (2.5f - 0.025f * L) * (0.116f * fabsf(sinf((h - 90.f) / 2.f * (float)M_PI / 180.f)) + 0.085f) * C;
  const float Yc = Yn * _fe_lab_fi((L + dL + 16.f) / 116.f);
  const float Yo = Yn * _fe_lab_fi((L + 16.f) / 116.f);
  if(Yo <= 1e-10f) return 1.f;
  return fminf(Yc / Yo, fe_hk_max_mul);
}

// ---------------------------------------------------------------------------
// Output gamut mapping -- replaces a naive per-channel [0,1] clamp with the
// same hue-preserving Ych/Yrg compression darktable's own filmicrgb.c uses
// (common/gamut_mapping.h), gamut-mapped against the module's own working
// space (linear Adobe RGB) rather than a foreign perceptual library: clip
// luminance to the valid reflectance range, then pull chroma in toward the
// Adobe RGB primaries' cube at constant hue, only falling back to a
// per-channel clamp as a final catch-all for whatever residual the chroma
// step (a "brute-force" bound, see Ych_max_chroma()'s own comment) leaves.
// ---------------------------------------------------------------------------
static inline void _fe_gamut_map(dt_aligned_pixel_t rgb,
                                 const dt_colormatrix_t input_matrix_trans,
                                 const dt_colormatrix_t output_matrix,
                                 const dt_colormatrix_t output_matrix_trans)
{
  for(int c = 0; c < 3; c++) rgb[c] = fmaxf(rgb[c], 0.f);

  dt_aligned_pixel_t Ych = { 0.f };
  RGB_to_Ych(rgb, input_matrix_trans, Ych);

  // Reflectance is defined on [0,1]; clip luminance to that range before
  // deriving the chroma bound below (which is only valid inside it).
  Ych[0] = CLAMPF(Ych[0], CIE_Y_1931_to_CIE_Y_2006(0.f), CIE_Y_1931_to_CIE_Y_2006(1.f));

  const float cos_h = Ych[2];
  const float sin_h = Ych[3];
  Ych[1] = fminf(Ych[1], Ych_max_chroma(output_matrix, 1.f, Ych[0], cos_h, sin_h));

  Ych_to_RGB(Ych, output_matrix_trans, rgb);

  // Final catch-all: chroma clipping above is a per-channel bound derived
  // independently for R, G, B (see Ych_max_chroma()'s own "brute-force"
  // comment), so one channel can still land a hair outside [0,1].
  for(int c = 0; c < 3; c++) rgb[c] = CLAMPF(rgb[c], 0.f, 1.f);
}

// ---------------------------------------------------------------------------
// commit_params: resolve the GUI selection into a fully-calibrated cascade
// per layer.
// ---------------------------------------------------------------------------
void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_filmemulation_params_t *p = (dt_iop_filmemulation_params_t *)p1;
  dt_iop_filmemulation_data_t *d = piece->data;

  d->use_hk = p->use_hk;

  if(p->type == DT_FE_TYPE_BW)
  {
    d->n_layers = 1;
    const dt_film_curve_t *filter = (p->bw_filter == DT_FE_FILTER_NONE) ? NULL : &fe_filters[p->bw_filter];
    const dt_film_curve_t *stages[FE_MAX_STAGES] = { &fe_trix_dev7, &fe_poly_curves[FE_POLY_GRADE_NORMAL], NULL };
    const gboolean increasing[FE_MAX_STAGES] = { TRUE, TRUE, FALSE };
    const float ref_d[FE_MAX_STAGES] = { fe_trix_ref_d, 0.f, 0.f };
    _fe_setup_layer(&d->layer[0], &fe_trix_sens, filter, 2, stages, increasing, ref_d, &fe_trix_fit, p->print_gamma);
  }
  else if(p->process == DT_FE_PROCESS_NEGATIVE)
  {
    d->n_layers = 3;
    const dt_film_negative_t *film = &fe_negative_films[p->negative_film];
    const dt_film_paper_t *paper = &fe_paper_ladder[p->negative_paper];
    for(int li = 0; li < 3; li++)
    {
      const dt_film_curve_t *stages[FE_MAX_STAGES] = { &film->curves[li], &paper->curves[li], NULL };
      const gboolean increasing[FE_MAX_STAGES] = { TRUE, TRUE, FALSE };
      const float ref_d[FE_MAX_STAGES] = { film->ref_d[li], 0.f, 0.f };
      _fe_setup_layer(&d->layer[li], &film->sens[li], NULL, 2, stages, increasing, ref_d, &film->fit[li], p->print_gamma);
    }
  }
  else if(p->route == DT_FE_ROUTE_DIRECT)
  {
    d->n_layers = 3;
    const dt_film_reversal_t *film = &fe_reversal_films[p->reversal_film];
    const dt_film_paper_t *paper = &fe_direct_print_papers[p->direct_paper];
    for(int li = 0; li < 3; li++)
    {
      const dt_film_curve_t *stages[FE_MAX_STAGES] = { &film->curves[li], &paper->curves[li], NULL };
      const gboolean increasing[FE_MAX_STAGES] = { FALSE, FALSE, FALSE };
      const float ref_d[FE_MAX_STAGES] = { film->ref_d[li], 0.f, 0.f };
      _fe_setup_layer(&d->layer[li], &film->sens[li], NULL, 2, stages, increasing, ref_d, &film->fit[li], p->print_gamma);
    }
  }
  else // reversal, internegative route
  {
    d->n_layers = 3;
    const dt_film_reversal_t *film = &fe_reversal_films[p->reversal_film];
    const dt_film_paper_t *paper = &fe_paper_ladder[p->internegative_paper];
    for(int li = 0; li < 3; li++)
    {
      const dt_film_curve_t *stages[FE_MAX_STAGES] = { &film->curves[li], &fe_internegative_curves[li], &paper->curves[li] };
      const gboolean increasing[FE_MAX_STAGES] = { FALSE, TRUE, TRUE };
      const float ref_d[FE_MAX_STAGES] = { film->ref_d[li], fe_internegative_lad_aim[li], 0.f };
      _fe_setup_layer(&d->layer[li], &film->sens[li], NULL, 3, stages, increasing, ref_d,
                      &fe_internegative_fits[li], p->print_gamma);
    }
  }
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_filmemulation_data_t));
}

// ---------------------------------------------------------------------------
// process: pipe RGB -> linear Adobe RGB primaries -> spectral reconstruction
// -> exposure integration per layer -> optional HK -> cascade -> hue-
// preserving gamut mapping (_fe_gamut_map()) -> linear Adobe RGB primaries
// -> pipe RGB.
// ---------------------------------------------------------------------------
void process(dt_iop_module_t *self,
            dt_dev_pixelpipe_iop_t *piece,
            const void *const ivoid,
            void *const ovoid,
            const dt_iop_roi_t *const roi_in,
            const dt_iop_roi_t *const roi_out)
{
  const dt_iop_filmemulation_data_t *d = piece->data;
  const float *const in = (const float *)ivoid;
  float *const out = (float *)ovoid;
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  const dt_iop_order_iccprofile_info_t *pipe_work_profile = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  const dt_iop_order_iccprofile_info_t *adobergb_profile
      = dt_ioppr_add_profile_info_to_list(self->dev, DT_COLORSPACE_ADOBERGB, "", DT_INTENT_RELATIVE_COLORIMETRIC);

  dt_colormatrix_t pipe_to_adobergb, adobergb_to_pipe;
  if(pipe_work_profile != NULL && adobergb_profile != NULL && pipe_work_profile != adobergb_profile)
  {
    dt_colormatrix_mul(pipe_to_adobergb, pipe_work_profile->matrix_in_transposed,
                       adobergb_profile->matrix_out_transposed);
    mat3SSEinv(adobergb_to_pipe, pipe_to_adobergb);
  }
  else
  {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 4; j++)
      {
        pipe_to_adobergb[i][j] = (i == j && i < 3) ? 1.f : 0.f;
        adobergb_to_pipe[i][j] = pipe_to_adobergb[i][j];
      }
  }

  // Matrices for the output gamut mapping (_fe_gamut_map()), gamut-mapped
  // against Adobe RGB since that is the space `result` is assembled in,
  // below, before the adobergb_to_pipe conversion. Falls back to the old
  // per-channel clamp in the vanishingly unlikely case the built-in Adobe
  // RGB profile info failed to generate.
  dt_colormatrix_t yrg_input_matrix_trans = { { 0.f } };
  dt_colormatrix_t yrg_output_matrix = { { 0.f } };
  dt_colormatrix_t yrg_output_matrix_trans = { { 0.f } };
  const gboolean has_gamut_matrices = (adobergb_profile != NULL);
  if(has_gamut_matrices)
  {
    dt_colormatrix_t yrg_input_matrix = { { 0.f } };
    prepare_RGB_Yrg_matrices(adobergb_profile, yrg_input_matrix, yrg_output_matrix);
    dt_colormatrix_transpose(yrg_input_matrix_trans, yrg_input_matrix);
    dt_colormatrix_transpose(yrg_output_matrix_trans, yrg_output_matrix);
  }

  const int n_layers = d->n_layers;
  const gboolean use_hk = d->use_hk;

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;

    dt_aligned_pixel_t rgb;
    dt_apply_transposed_color_matrix(pix_in, pipe_to_adobergb, rgb);

    float coeffs[3];
    _fe_rgb_to_sigmoid_coeffs(rgb[0], rgb[1], rgb[2], coeffs);
    float spectrum[FE_N_WAVELENGTHS];
    _fe_sigmoid_spectrum(coeffs, spectrum);

    const float hk = use_hk ? _fe_hk_mul(rgb[0], rgb[1], rgb[2]) : 1.f;

    dt_aligned_pixel_t result = { 0.f, 0.f, 0.f, 0.f };
    for(int layer = 0; layer < n_layers; layer++)
    {
      float E = 0.f;
      for(int w = 0; w < FE_N_WAVELENGTHS; w++) E += spectrum[w] * d->layer[layer].weight[w];
      E *= hk;
      result[layer] = _fe_xfer(&d->layer[layer], E);
    }
    if(n_layers == 1)
    {
      result[1] = result[0];
      result[2] = result[0];
    }
    result[3] = pix_in[3];

    if(has_gamut_matrices)
      _fe_gamut_map(result, yrg_input_matrix_trans, yrg_output_matrix, yrg_output_matrix_trans);
    else
      for(int c = 0; c < 3; c++) result[c] = fminf(fmaxf(result[c], 0.f), 1.f);

    dt_apply_transposed_color_matrix(result, adobergb_to_pipe, pix_out);
  }
}

// ---------------------------------------------------------------------------
// GUI: cascading dropdowns. All widgets are built once in gui_init() and
// grouped into labeled boxes; gui_changed() shows/hides whole boxes based on
// the current type/process/route selection (same pattern as sigmoid.c's
// primaries_section/hue_preservation_slider and denoiseprofile.c's per-mode
// widget groups).
// ---------------------------------------------------------------------------
void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  const dt_iop_filmemulation_gui_data_t *g = self->gui_data;
  const dt_iop_filmemulation_params_t *p = self->params;

  if(!w || w == g->type)
  {
    gtk_widget_set_visible(g->box_bw, p->type == DT_FE_TYPE_BW);
    gtk_widget_set_visible(g->box_color, p->type == DT_FE_TYPE_COLOR);
  }
  if(!w || w == g->process)
  {
    gtk_widget_set_visible(g->box_negative, p->process == DT_FE_PROCESS_NEGATIVE);
    gtk_widget_set_visible(g->box_reversal, p->process == DT_FE_PROCESS_REVERSAL);
  }
  if(!w || w == g->route)
  {
    gtk_widget_set_visible(g->box_direct, p->route == DT_FE_ROUTE_DIRECT);
    gtk_widget_set_visible(g->box_internegative, p->route == DT_FE_ROUTE_INTERNEGATIVE);
  }
}

void gui_update(dt_iop_module_t *self)
{
  gui_changed(self, NULL, NULL);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_filmemulation_gui_data_t *g = IOP_GUI_ALLOC(filmemulation);

  g->type = dt_bauhaus_combobox_from_params(self, "type");
  // self->widget is lazily created (as a vbox) by the first
  // dt_bauhaus_*_from_params() call above -- capture it only now.
  GtkWidget *main_box = self->widget;

  g->box_bw = self->widget = dt_gui_vbox();
  g->bw_film = dt_bauhaus_combobox_from_params(self, "bw_film");
  g->bw_filter = dt_bauhaus_combobox_from_params(self, "bw_filter");
  gtk_widget_set_tooltip_text(g->bw_filter, _("Wratten glass filter -- folded into the film's spectral response"));
  dt_gui_box_add(main_box, g->box_bw);

  g->box_color = self->widget = dt_gui_vbox();
  g->process = dt_bauhaus_combobox_from_params(self, "process");

  g->box_negative = self->widget = dt_gui_vbox();
  g->negative_film = dt_bauhaus_combobox_from_params(self, "negative_film");
  g->negative_paper = dt_bauhaus_combobox_from_params(self, "negative_paper");
  dt_gui_box_add(g->box_color, g->box_negative);

  g->box_reversal = self->widget = dt_gui_vbox();
  g->reversal_film = dt_bauhaus_combobox_from_params(self, "reversal_film");
  g->route = dt_bauhaus_combobox_from_params(self, "route");
  gtk_widget_set_tooltip_text(g->route, _("straight to paper: a real print paper built to accept a reversal\n"
                                          "original directly. internegative: duplicated onto a real\n"
                                          "duplicating internegative first, then printed like an ordinary\n"
                                          "negative -- what most real darkroom labs did for a slide."));

  g->box_direct = self->widget = dt_gui_vbox();
  g->direct_paper = dt_bauhaus_combobox_from_params(self, "direct_paper");
  dt_gui_box_add(g->box_reversal, g->box_direct);

  g->box_internegative = self->widget = dt_gui_vbox();
  g->internegative_paper = dt_bauhaus_combobox_from_params(self, "internegative_paper");
  dt_gui_box_add(g->box_reversal, g->box_internegative);

  dt_gui_box_add(g->box_color, g->box_reversal);
  dt_gui_box_add(main_box, g->box_color);

  self->widget = main_box;

  g->print_gamma = dt_bauhaus_slider_from_params(self, "print_gamma");
  gtk_widget_set_tooltip_text(g->print_gamma, _("Jones system-gamma target: how contrasty the film+paper pairing\n"
                                          "renders overall (\"punchiness\"). 1.0 is colorimetrically faithful;\n"
                                          "real print materials are usually viewed punchier than that."));
  g->use_hk = dt_bauhaus_toggle_from_params(self, "use_hk");
  gtk_widget_set_tooltip_text(g->use_hk, _("account for saturated colors appearing brighter than their\n"
                                           "measured luminance (Helmholtz-Kohlrausch effect)"));
}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-space on;
