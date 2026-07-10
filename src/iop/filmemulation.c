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
#include "common/gaussian.h"
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
#include <string.h>

#include "external/filmemulation_data.c"
#include "external/filmemulation_data_spektrafilm.c"

DT_MODULE_INTROSPECTION(2, dt_iop_filmemulation_params_t)

// ---------------------------------------------------------------------------
// Params enums
// ---------------------------------------------------------------------------

// Which film/paper data set this instance of the module draws from. The
// "initial version" data set (fe_* tables in external/filmemulation_data.c)
// is darktable's own port, built around a live gamma-correction fit; the
// "spektrafilm" data set (dt_sf_* tables in
// external/filmemulation_data_spektrafilm.c) is a straight port of the
// spektrafilm project's (https://github.com/andreavolpato/spektrafilm)
// own film/paper profiles *and* its own rendering algorithm (per-stock
// Hanatos2025 spectral-sensitivity adaptation, direct H&D curve lookup,
// a real film->paper spectral printing cascade through an enlarger
// illuminant, and its parametric print density-curve model) -- see that
// project's own README for the darkroom/color-science rationale.
typedef enum dt_iop_filmemulation_dataset_t
{
  DT_FE_DATASET_INITIAL = 0,     // $DESCRIPTION: "initial version"
  DT_FE_DATASET_SPEKTRAFILM = 1, // $DESCRIPTION: "spektrafilm"
} dt_iop_filmemulation_dataset_t;

// Film/paper stock selectors for spektrafilm mode -- one enum constant per
// bundled profile (see dt_sf_films/dt_sf_papers in
// external/filmemulation_data_spektrafilm.c), values 0..19/0..7 matching
// those arrays' order exactly. Defined here rather than in that #include'd
// data file because darktable's introspection generator (tools/
// introspection/*.pl) doesn't pick up enum typedefs from files this one
// #include's -- dt_bauhaus_combobox_from_params() needs these visible
// directly in filmemulation.c, like every other params enum here.
typedef enum dt_iop_filmemulation_sf_film_t
{
  DT_SF_FILM_FUJIFILM_C200 = 0, // $DESCRIPTION: "Fujifilm C200"
  DT_SF_FILM_FUJIFILM_PRO_400H = 1, // $DESCRIPTION: "Fujifilm Pro 400H"
  DT_SF_FILM_FUJIFILM_PROVIA_100F = 2, // $DESCRIPTION: "Fujifilm Provia 100F"
  DT_SF_FILM_FUJIFILM_VELVIA_100 = 3, // $DESCRIPTION: "Fujifilm Velvia 100"
  DT_SF_FILM_FUJIFILM_XTRA_400 = 4, // $DESCRIPTION: "Fujifilm X-Tra 400"
  DT_SF_FILM_KODAK_EKTACHROME_100 = 5, // $DESCRIPTION: "Kodak Ektachrome 100"
  DT_SF_FILM_KODAK_EKTAR_100 = 6, // $DESCRIPTION: "Kodak Ektar 100"
  DT_SF_FILM_KODAK_GOLD_200 = 7, // $DESCRIPTION: "Kodak Gold 200"
  DT_SF_FILM_KODAK_KODACHROME_64 = 8, // $DESCRIPTION: "Kodak Kodachrome 64"
  DT_SF_FILM_KODAK_PORTRA_160 = 9, // $DESCRIPTION: "Kodak Portra 160"
  DT_SF_FILM_KODAK_PORTRA_400 = 10, // $DESCRIPTION: "Kodak Portra 400"
  DT_SF_FILM_KODAK_PORTRA_800 = 11, // $DESCRIPTION: "Kodak Portra 800"
  DT_SF_FILM_KODAK_PORTRA_800_PUSH1 = 12, // $DESCRIPTION: "Kodak Portra 800 (Push 1)"
  DT_SF_FILM_KODAK_PORTRA_800_PUSH2 = 13, // $DESCRIPTION: "Kodak Portra 800 (Push 2)"
  DT_SF_FILM_KODAK_ULTRAMAX_400 = 14, // $DESCRIPTION: "Kodak Ultramax 400"
  DT_SF_FILM_KODAK_VERITA_200D = 15, // $DESCRIPTION: "Kodak Verita 200D"
  DT_SF_FILM_KODAK_VISION3_200T = 16, // $DESCRIPTION: "Kodak Vision3 200T"
  DT_SF_FILM_KODAK_VISION3_250D = 17, // $DESCRIPTION: "Kodak Vision3 250D"
  DT_SF_FILM_KODAK_VISION3_500T = 18, // $DESCRIPTION: "Kodak Vision3 500T"
  DT_SF_FILM_KODAK_VISION3_50D = 19, // $DESCRIPTION: "Kodak Vision3 50D"
} dt_iop_filmemulation_sf_film_t;

typedef enum dt_iop_filmemulation_sf_paper_t
{
  DT_SF_PAPER_FUJIFILM_CRYSTAL_ARCHIVE_TYPEII = 0, // $DESCRIPTION: "Fujifilm Crystal Archive Type II"
  DT_SF_PAPER_KODAK_2383 = 1, // $DESCRIPTION: "Kodak Vision 2383"
  DT_SF_PAPER_KODAK_2393 = 2, // $DESCRIPTION: "Kodak Vision Premier 2393"
  DT_SF_PAPER_KODAK_EKTACOLOR_EDGE = 3, // $DESCRIPTION: "Kodak Ektacolor Edge"
  DT_SF_PAPER_KODAK_ENDURA_PREMIER = 4, // $DESCRIPTION: "Kodak Professional Endura Premier"
  DT_SF_PAPER_KODAK_PORTRA_ENDURA = 5, // $DESCRIPTION: "Kodak Professional Portra Endura"
  DT_SF_PAPER_KODAK_SUPRA_ENDURA = 6, // $DESCRIPTION: "Kodak Professional Supra Endura"
  DT_SF_PAPER_KODAK_ULTRA_ENDURA = 7, // $DESCRIPTION: "Kodak Professional Ultra Endura"
} dt_iop_filmemulation_sf_paper_t;

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

// How a normalized pipe RGB is turned into the per-wavelength spectrum that
// gets dotted against each layer's spectral sensitivity -- three published
// "spectral upsampling" methods, all recovering a plausible but necessarily
// arbitrary metamer for the same RGB (see task 08's own writeup for why this
// is inherently a guess, and where the three differ in practice).
typedef enum dt_iop_filmemulation_spectral_method_t
{
  // Jakob & Hanika 2019: an analytic sigmoid whose 3 coefficients come from
  // a baked LUT fitted for smoothness -- darktable's original method here.
  DT_FE_SPECTRAL_JAKOB_HANIKA2019 = 0, // $DESCRIPTION: "smooth sigmoid (Jakob & Hanika 2019)"
  // Mallett & Yuksel 2019: a fixed 3-spectrum linear reflectance basis --
  // simpler and cheaper, not more accurate than Jakob-Hanika.
  DT_FE_SPECTRAL_MALLETT2019 = 1, // $DESCRIPTION: "linear basis (Mallett & Yuksel 2019)"
  // Hanatos 2025: a closed-form reconstruction fitted against real curated/
  // measured spectra rather than an idealized smoothness criterion -- can
  // diverge from the other two for narrow-band/heavily colored light.
  DT_FE_SPECTRAL_HANATOS2025 = 2, // $DESCRIPTION: "measured-spectra fit (Hanatos 2025)"
} dt_iop_filmemulation_spectral_method_t;

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
  dt_iop_filmemulation_dataset_t dataset; // $DEFAULT: DT_FE_DATASET_INITIAL $DESCRIPTION: "data set"

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

  dt_iop_filmemulation_spectral_method_t spectral_method; // $DEFAULT: DT_FE_SPECTRAL_HANATOS2025 $DESCRIPTION: "spectral reconstruction"

  float print_gamma; // $MIN: 1.0 $MAX: 1.6 $DEFAULT: 1.25 $DESCRIPTION: "print gamma"
  gboolean use_hk; // $DEFAULT: FALSE $DESCRIPTION: "perceptual color correction (HK)"

  gboolean dir_couplers_active; // $DEFAULT: TRUE $DESCRIPTION: "dye coupler interaction"
  float dir_couplers_amount; // $MIN: 0.0 $MAX: 1.5 $DEFAULT: 1.0 $DESCRIPTION: "amount"

  // spektrafilm data set only (dataset == DT_FE_DATASET_SPEKTRAFILM); the
  // dir_couplers_* fields above are shared with the initial-version data
  // set and apply here too (same dye-coupler-crosstalk knob, different
  // underlying matrix -- see dt_sf_film_t.couplers_matrix).
  dt_iop_filmemulation_sf_film_t sf_film;   // $DEFAULT: DT_SF_FILM_KODAK_PORTRA_400 $DESCRIPTION: "film"
  dt_iop_filmemulation_sf_paper_t sf_paper; // $DEFAULT: DT_SF_PAPER_KODAK_PORTRA_ENDURA $DESCRIPTION: "paper"
  float sf_camera_ev; // $MIN: -3.0 $MAX: 3.0 $DEFAULT: 0.0 $DESCRIPTION: "camera exposure"
  float sf_print_exposure; // $MIN: 0.1 $MAX: 4.0 $DEFAULT: 1.0 $DESCRIPTION: "print exposure"
  float sf_print_gamma; // $MIN: 0.5 $MAX: 2.0 $DEFAULT: 1.0 $DESCRIPTION: "print gamma"
  gboolean sf_hanatos_surface; // $DEFAULT: FALSE $DESCRIPTION: "per-stock spectral correction"
} dt_iop_filmemulation_params_t;

// v1 -> v2: added the spektrafilm data set (the "dataset" selector plus its
// own sf_* fields). dataset was inserted as the new struct's first member,
// so this is a field-by-field remap rather than a simple memcpy-prefix.
int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  typedef struct dt_iop_filmemulation_params_v2_t
  {
    dt_iop_filmemulation_dataset_t dataset;
    dt_iop_filmemulation_type_t type;
    dt_iop_filmemulation_bw_film_t bw_film;
    dt_iop_filmemulation_bw_filter_t bw_filter;
    dt_iop_filmemulation_process_t process;
    dt_iop_filmemulation_reversal_film_t reversal_film;
    dt_iop_filmemulation_route_t route;
    dt_iop_filmemulation_direct_paper_t direct_paper;
    dt_iop_filmemulation_ladder_paper_t internegative_paper;
    dt_iop_filmemulation_negative_film_t negative_film;
    dt_iop_filmemulation_ladder_paper_t negative_paper;
    dt_iop_filmemulation_spectral_method_t spectral_method;
    float print_gamma;
    gboolean use_hk;
    gboolean dir_couplers_active;
    float dir_couplers_amount;
    dt_iop_filmemulation_sf_film_t sf_film;
    dt_iop_filmemulation_sf_paper_t sf_paper;
    float sf_camera_ev;
    float sf_print_exposure;
    float sf_print_gamma;
    gboolean sf_hanatos_surface;
  } dt_iop_filmemulation_params_v2_t;

  if(old_version == 1)
  {
    typedef struct dt_iop_filmemulation_params_v1_t
    {
      dt_iop_filmemulation_type_t type;
      dt_iop_filmemulation_bw_film_t bw_film;
      dt_iop_filmemulation_bw_filter_t bw_filter;
      dt_iop_filmemulation_process_t process;
      dt_iop_filmemulation_reversal_film_t reversal_film;
      dt_iop_filmemulation_route_t route;
      dt_iop_filmemulation_direct_paper_t direct_paper;
      dt_iop_filmemulation_ladder_paper_t internegative_paper;
      dt_iop_filmemulation_negative_film_t negative_film;
      dt_iop_filmemulation_ladder_paper_t negative_paper;
      dt_iop_filmemulation_spectral_method_t spectral_method;
      float print_gamma;
      gboolean use_hk;
      gboolean dir_couplers_active;
      float dir_couplers_amount;
    } dt_iop_filmemulation_params_v1_t;

    const dt_iop_filmemulation_params_v1_t *old = (const dt_iop_filmemulation_params_v1_t *)old_params;
    dt_iop_filmemulation_params_v2_t *new = calloc(1, sizeof(dt_iop_filmemulation_params_v2_t));

    new->dataset = DT_FE_DATASET_INITIAL;
    new->type = old->type;
    new->bw_film = old->bw_film;
    new->bw_filter = old->bw_filter;
    new->process = old->process;
    new->reversal_film = old->reversal_film;
    new->route = old->route;
    new->direct_paper = old->direct_paper;
    new->internegative_paper = old->internegative_paper;
    new->negative_film = old->negative_film;
    new->negative_paper = old->negative_paper;
    new->spectral_method = old->spectral_method;
    new->print_gamma = old->print_gamma;
    new->use_hk = old->use_hk;
    new->dir_couplers_active = old->dir_couplers_active;
    new->dir_couplers_amount = old->dir_couplers_amount;
    new->sf_film = DT_SF_FILM_KODAK_PORTRA_400;
    new->sf_paper = DT_SF_PAPER_KODAK_PORTRA_ENDURA;
    new->sf_camera_ev = 0.0f;
    new->sf_print_exposure = 1.0f;
    new->sf_print_gamma = 1.0f;
    new->sf_hanatos_surface = FALSE;

    *new_params = new;
    *new_params_size = sizeof(dt_iop_filmemulation_params_v2_t);
    *new_version = 2;
    return 0;
  }
  return 1;
}

typedef struct dt_iop_filmemulation_gui_data_t
{
  GtkWidget *dataset;
  GtkWidget *box_initial; // everything from the "initial version" data set
  GtkWidget *type;
  GtkWidget *box_bw, *bw_film, *bw_filter;
  GtkWidget *box_color, *process;
  GtkWidget *box_negative, *negative_film, *negative_paper;
  GtkWidget *box_reversal, *reversal_film, *route;
  GtkWidget *box_direct, *direct_paper;
  GtkWidget *box_internegative, *internegative_paper;
  GtkWidget *spectral_method;
  GtkWidget *print_gamma, *use_hk;
  GtkWidget *box_spektrafilm; // everything from the "spektrafilm" data set
  GtkWidget *sf_film, *sf_paper;
  GtkWidget *sf_camera_ev, *sf_print_exposure, *sf_print_gamma, *sf_hanatos_surface;
  GtkWidget *dir_couplers_active, *dir_couplers_amount; // shared by both data sets
} dt_iop_filmemulation_gui_data_t;

// ---------------------------------------------------------------------------
// Pipeline (piece->data) representation of a resolved cascade -- one entry
// per color layer (1 for B&W, 3 for color). All the "search"/calibration
// work (_fe_find_anchor, gamma-correction factor derivation) happens once
// here, in commit_params(); the per-pixel hot path (_fe_xfer()) is just a
// couple of small interpolations/closed-form evaluations.
// ---------------------------------------------------------------------------
#define FE_MAX_STAGES 3

// DIR-coupler spatial diffusion constants -- global across all stocks
// (spektrafilm's DirCouplersParams defaults; no stock overrides these).
// Gaussian core + a second, wider Gaussian standing in for the exponential
// long-range tail, blended by weight (same Gaussian-mixture-for-exponential
// approximation as the halation task).
#define FE_DIRC_DIFFUSION_UM 20.0f
#define FE_DIRC_DIFFUSION_TAIL_UM 200.0f
#define FE_DIRC_DIFFUSION_TAIL_WEIGHT 0.06f
// Assumed 35mm-film physical width (in microns) mapped to the image width,
// used to convert the _UM constants above into pixel-space sigmas so the
// diffusion radius scales with output resolution rather than being a fixed
// pixel count.
#define FE_FILM_WIDTH_UM 36000.0f

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

// ---------------------------------------------------------------------------
// spektrafilm-mode resolved state (dataset == DT_FE_DATASET_SPEKTRAFILM) --
// a from-scratch port of spektrafilm's own SimulationPipeline: RGB -> per-
// stock Hanatos2025 spectral upsampling -> film exposure/density (direct
// H&D curve lookup, DIR-coupler correction) -> film->paper printing
// cascade through an enlarger illuminant -> paper's parametric print
// density-curve model -> paper density -> viewing-illuminant spectral scan
// -> display RGB. Everything here is resolved once in commit_params()
// (_sf_commit_params()); process()'s per-pixel path (_sf_process_*()) is
// pure evaluation of already-baked film/paper data (filmemulation_data_
// spektrafilm.c) plus this struct's setup-time-only derived constants.
// ---------------------------------------------------------------------------
typedef struct dt_iop_filmemulation_sf_data_t
{
  gboolean active; // dataset == DT_FE_DATASET_SPEKTRAFILM
  const dt_sf_film_t *film;
  const dt_sf_paper_t *paper;
  gboolean surface_active; // apply the per-stock Hanatos2025 "surface" log-exposure correction
  float camera_exposure_mul; // 2^sf_camera_ev
  float print_exposure; // sf_print_exposure slider
  float print_gamma; // sf_print_gamma slider -- s023 print-curve morph's (uniform) gamma_factor

  gboolean dir_couplers_active;
  float dir_couplers_matrix[3][3]; // film->couplers_matrix scaled by dir_couplers_amount
  // Curves the film would need to have, before DIR-coupler self/cross-
  // inhibition, to reproduce its own real (measured) density_curves after
  // the coupler correction is applied -- see
  // _sf_setup_dir_couplers_curves()/compute_density_curves_before_dir_
  // couplers(). Only populated (and only used) when dir_couplers_active.
  float density_curves_0[SF_N_LOG_EXPOSURE][3];

  float enlarger_light[SF_N_WAVELENGTHS]; // TH-KG3 lamp, filtered by this film+paper pair's neutral CC dichroics
  float exposure_factor; // print-exposure normalizer: 1/geomean(film-midgray raw through enlarger+paper sensitivity)

  const float *view_illuminant; // paper's own viewing illuminant spectrum (sf_view_illuminants[paper->viewing_illuminant])
  float view_norm; // sum(view_illuminant * ybar) -- scanning-stage XYZ normalization
  const float (*view_to_adobergb)[3]; // sf_view_illuminant_to_adobergb[paper->viewing_illuminant]

  // Positive (reversal/slide) film only: real photography views a
  // developed transparency directly rather than printing it onto
  // negative-working paper (which would invert it) -- matches
  // spektrafilm's io.scan_film branch in SimulationPipeline._build_
  // topology(), specialized here to always scan when the film itself is
  // positive rather than exposing it as a separate user setting. Same
  // trio of fields as above, sourced from the film's own viewing
  // illuminant/density response instead of the paper's.
  const float *film_view_illuminant;
  float film_view_norm;
  const float (*film_view_to_adobergb)[3];
} dt_iop_filmemulation_sf_data_t;

typedef struct dt_iop_filmemulation_data_t
{
  int n_layers; // 1 (B&W) or 3 (color)
  dt_iop_filmemulation_layer_data_t layer[3];
  gboolean use_hk;

  dt_iop_filmemulation_spectral_method_t spectral_method;
  // Hanatos2025 only: the reconstructed spectral shape at the working
  // white point, pre-scaled by white's own XYZ sum -- every pixel's
  // reconstructed spectrum is divided by this so a neutral RGB reconstructs
  // as a flat reflectance of that same value, exactly like the other two
  // methods (see _fe_hanatos_spectrum()).
  float hanatos_ref_shape[FE_N_WAVELENGTHS];

  // DIR-coupler (dye coupler inter-layer crosstalk) state -- only ever
  // active for n_layers == 3 (color). See _fe_setup_dir_couplers() and
  // the spatial diffusion pass in process().
  gboolean dir_couplers_active;
  gboolean dir_couplers_positive; // TRUE for reversal (any route), FALSE for negative
  float dir_couplers_matrix[3][3]; // [donor][receiver], pre-scaled by the amount slider
  float dir_couplers_density_max[3]; // per-layer stage-0 max density (for the positive/silver term)

  dt_iop_filmemulation_sf_data_t sf;
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

// ---------------------------------------------------------------------------
// DIR-coupler (dye coupler inter-layer crosstalk) setup -- direct port of
// Python's compute_dir_couplers_matrix() plus a darktable-specific
// grey-anchor fixed-point correction (see task 02's task file for the
// derivation) that keeps a flat 18% grey field's rendering exactly
// unaffected by the coupler correction -- the same invariant Python's
// compute_density_curves_before_dir_couplers() enforces via a curve
// resample, reformulated here as a fixed point on na0 so it works
// uniformly whether stage 0 is the analytically-fitted "corrected" stage
// (negative, reversal-direct) or a plain interpolated curve (reversal
// internegative route), since both go through the same _fe_eval_stage().
// ---------------------------------------------------------------------------
static void _fe_build_dir_couplers_matrix(const dt_film_dir_couplers_t *c, const float amount, float M[3][3])
{
  for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) M[i][j] = 0.f;
  M[0][0] = c->gamma_samelayer_rgb[0];
  M[1][1] = c->gamma_samelayer_rgb[1];
  M[2][2] = c->gamma_samelayer_rgb[2];
  M[0][1] = c->gamma_interlayer_r_to_gb[0]; // R -> G
  M[0][2] = c->gamma_interlayer_r_to_gb[1]; // R -> B
  M[1][0] = c->gamma_interlayer_g_to_rb[0]; // G -> R
  M[1][2] = c->gamma_interlayer_g_to_rb[1]; // G -> B
  M[2][0] = c->gamma_interlayer_b_to_rg[0]; // B -> R
  M[2][1] = c->gamma_interlayer_b_to_rg[1]; // B -> G
  for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) M[i][j] *= amount;
}

// Max density of a layer's stage-0 curve -- the analytic split-gauss fit's
// own asymptote when stage 0 is the print-gamma-corrected stage (negative,
// reversal-direct), or the digitized curve's own max y otherwise (reversal
// internegative route).
static float _fe_stage0_max_density(const dt_iop_filmemulation_layer_data_t *L)
{
  if(L->corrected_stage == 0)
    return fmaxf(L->fit.d_lo, L->fit.d_hi);
  const dt_film_curve_point_t *pts = L->stage[0].pts;
  const int n = L->stage[0].n;
  float m = pts[0].y;
  for(int i = 1; i < n; i++) if(pts[i].y > m) m = pts[i].y;
  return m;
}

// Re-derive each layer's na0 (the grey-exposure anchor -- see
// _fe_setup_layer()) so that a spatially-uniform 18% grey field renders
// identically whether DIR couplers are active or not: on a flat field the
// spatial diffusion in process() is a no-op, so the per-pixel correction
// there reduces to exactly the matrix term computed here. Fixed-point
// iterated (a handful of iterations is enough given gammas < 1 and smooth,
// bounded stage-0 responses) rather than solved in closed form, since it
// has to hold across all three cross-coupled layers at once.
static void _fe_setup_dir_couplers_anchor(dt_iop_filmemulation_data_t *d)
{
  float na0_base[3];
  for(int li = 0; li < 3; li++) na0_base[li] = d->layer[li].na0;

  float na0_new[3];
  for(int li = 0; li < 3; li++) na0_new[li] = na0_base[li];

  for(int iter = 0; iter < 8; iter++)
  {
    float D0[3];
    for(int li = 0; li < 3; li++)
      D0[li] = _fe_eval_stage(&d->layer[li], 0, na0_new[li]);

    float silver[3];
    for(int li = 0; li < 3; li++)
      silver[li] = d->dir_couplers_positive ? (d->dir_couplers_density_max[li] - D0[li]) : D0[li];

    float correction[3] = { 0.f, 0.f, 0.f };
    for(int receiver = 0; receiver < 3; receiver++)
      for(int donor = 0; donor < 3; donor++)
        correction[receiver] += silver[donor] * d->dir_couplers_matrix[donor][receiver];

    for(int li = 0; li < 3; li++) na0_new[li] = na0_base[li] + correction[li];
  }

  for(int li = 0; li < 3; li++) d->layer[li].na0 = na0_new[li];
}

// Called once per commit_params(), after all three layers' _fe_setup_layer()
// have run, for the three color branches (negative, reversal-direct,
// reversal-internegative) -- never for B&W, which has no inter-layer
// crosstalk to model.
static void _fe_configure_dir_couplers(dt_iop_filmemulation_data_t *d,
                                       const dt_iop_filmemulation_params_t *p,
                                       const dt_film_dir_couplers_t *couplers,
                                       const gboolean positive)
{
  d->dir_couplers_active = p->dir_couplers_active;
  d->dir_couplers_positive = positive;
  if(!d->dir_couplers_active) return;

  _fe_build_dir_couplers_matrix(couplers, p->dir_couplers_amount, d->dir_couplers_matrix);
  for(int li = 0; li < 3; li++)
    d->dir_couplers_density_max[li] = _fe_stage0_max_density(&d->layer[li]);
  _fe_setup_dir_couplers_anchor(d);
}

// Linear exposure E -> the log-position "lh" fed to stage 0 -- split out
// of _fe_xfer() so DIR-coupler handling (below) can compute this once,
// look at stage 0's first-pass density, then re-run the cascade from a
// corrected lh without duplicating the E->lh formula.
static inline float _fe_lh_from_E(const dt_iop_filmemulation_layer_data_t *L, const float E)
{
  return (E <= 1e-9f) ? L->x0_sentinel : (L->na0 + log10f(E / fe_grey));
}

// Full cascade transfer function starting from an already-computed lh --
// matches Python's build_print_cascade()'s inner xfer() closure exactly.
// Deliberately NOT clamped to [0,1] here: a highlight can legitimately
// reconstruct to a reflectance above 1 (or, per-channel, produce an
// out-of-gamut chromaticity once the three layers are recombined) and
// chopping each channel independently right here is exactly the naive
// per-channel clamp _fe_gamut_map() below replaces with a hue-preserving
// gamut compression once all three layers are assembled.
static inline float _fe_xfer_from_lh(const dt_iop_filmemulation_layer_data_t *L, const float lh)
{
  float D = _fe_eval_stage(L, 0, lh);
  for(int i = 1; i < L->n_stages; i++)
    D = _fe_eval_stage(L, i, L->pls[i - 1] - D);
  const float v = powf(10.f, -(D - L->fdm));
  return fmaxf(v, 0.f);
}

static inline float _fe_xfer(const dt_iop_filmemulation_layer_data_t *L, const float E)
{
  return _fe_xfer_from_lh(L, _fe_lh_from_E(L, E));
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
// Spectral reconstruction, alternate method 1: Mallett & Yuksel 2019 -- a
// fixed linear reflectance basis (fe_mallett_basis_{r,g,b} in
// filmemulation_data.c). By construction the three basis spectra sum to
// 1.0 at every wavelength, so a neutral RGB reconstructs as a flat
// reflectance of that same value automatically, matching how
// _fe_sigmoid_spectrum() above handles a neutral input.
// ---------------------------------------------------------------------------
static inline void _fe_mallett_spectrum(const float rgb[3], float spectrum[FE_N_WAVELENGTHS])
{
  const float r = CLAMPF(rgb[0], 0.f, 1.f);
  const float g = CLAMPF(rgb[1], 0.f, 1.f);
  const float b = CLAMPF(rgb[2], 0.f, 1.f);
  for(int i = 0; i < FE_N_WAVELENGTHS; i++)
    spectrum[i] = r * fe_mallett_basis_r[i] + g * fe_mallett_basis_g[i] + b * fe_mallett_basis_b[i];
}

// ---------------------------------------------------------------------------
// Spectral reconstruction, alternate method 2: Hanatos 2025 -- a closed-form
// rational-function fit against real curated/measured spectra (rather than
// an idealized smoothness criterion), indexed by chromaticity via a baked
// coefficient LUT (fe_hanatos_coeffs in filmemulation_data.c, itself a
// downsampled port of spektrafilm's hanatos_irradiance_xy_coeffs LUT).
// ---------------------------------------------------------------------------

// Maps a CIE xy chromaticity into the LUT's square-warped grid coordinate
// -- matches spektrafilm's _tri2quad(): samples the LUT with uniform
// density across the whole xy triangle instead of wasting resolution
// outside the visible locus.
static inline void _fe_tri2quad(const float cx, const float cy, float *tx, float *ty)
{
  const float omc = fmaxf(1.f - cx, 1e-6f);
  *ty = CLAMPF(cy / omc, 0.f, 1.f);
  *tx = CLAMPF(omc * omc, 0.f, 1.f);
}

static inline void _fe_hanatos_fetch_coeffs(const float tx, const float ty, float coeffs[4])
{
  const int N = FE_HANATOS_GRID_SIZE;
  const float gx = CLAMPF(tx, 0.f, 1.f) * (N - 1);
  const float gy = CLAMPF(ty, 0.f, 1.f) * (N - 1);
  int x0 = (int)gx; if(x0 > N - 2) x0 = N - 2;
  int y0 = (int)gy; if(y0 > N - 2) y0 = N - 2;
  const float fx = gx - x0, fy = gy - y0;
  for(int k = 0; k < 4; k++)
  {
    const float c00 = fe_hanatos_coeffs[x0][y0][k];
    const float c01 = fe_hanatos_coeffs[x0][y0 + 1][k];
    const float c10 = fe_hanatos_coeffs[x0 + 1][y0][k];
    const float c11 = fe_hanatos_coeffs[x0 + 1][y0 + 1][k];
    const float lo = c00 * (1.f - fy) + c01 * fy;
    const float hi = c10 * (1.f - fy) + c11 * fy;
    coeffs[k] = lo * (1.f - fx) + hi * fx;
  }
}

// Reconstructs the Hanatos-2025 spectrum shape for an RGB color, scaled by
// its own XYZ magnitude -- matches spektrafilm's compute_lut_spectra()'s
// per-pixel evaluation (_fetch_coeffs() + _compute_spectra_from_coeffs()),
// collapsed to a direct per-wavelength evaluation instead of the Python
// reference's upsample/blur/downsample (that resampling exists there only
// to anti-alias before interpolating onto a coarser measurement grid; the
// rational-function shape itself is smooth enough to evaluate directly at
// darktable's own 31 wavelengths). Not clamped to reflectance range: this
// represents a full scene-irradiance-like spectrum, not a reflectance, so
// its overall scale is normalized away by the caller instead (see
// _fe_hanatos_spectrum() and hanatos_ref_shape's own comment).
static inline void _fe_hanatos_reconstruct(const float rgb[3], float shape_scaled[FE_N_WAVELENGTHS])
{
  const float r = fmaxf(rgb[0], 0.f), g = fmaxf(rgb[1], 0.f), b = fmaxf(rgb[2], 0.f);
  const float X = fe_adobergb_rgb2xyz[0][0] * r + fe_adobergb_rgb2xyz[0][1] * g + fe_adobergb_rgb2xyz[0][2] * b;
  const float Y = fe_adobergb_rgb2xyz[1][0] * r + fe_adobergb_rgb2xyz[1][1] * g + fe_adobergb_rgb2xyz[1][2] * b;
  const float Z = fe_adobergb_rgb2xyz[2][0] * r + fe_adobergb_rgb2xyz[2][1] * g + fe_adobergb_rgb2xyz[2][2] * b;
  const float sum = fmaxf(X + Y + Z, 1e-9f);

  float tx, ty;
  _fe_tri2quad(X / sum, Y / sum, &tx, &ty);

  float coeffs[4];
  _fe_hanatos_fetch_coeffs(tx, ty, coeffs);
  const float c3 = fmaxf(coeffs[3], 1e-6f);

  for(int i = 0; i < FE_N_WAVELENGTHS; i++)
  {
    const float wl = fe_wavelengths[i];
    const float x = (coeffs[0] * wl + coeffs[1]) * wl + coeffs[2];
    const float y = 1.f / sqrtf(x * x + 1.f);
    const float shape = (0.5f * x * y + 0.5f) / c3;
    shape_scaled[i] = shape * sum;
  }
}

// Per-pixel Hanatos-2025 spectrum: the reconstructed shape above, divided
// by the same shape reconstructed for the working white point (see
// hanatos_ref_shape's own comment in dt_iop_filmemulation_data_t) -- puts
// it on the same "flat reflectance of 1 under white light" footing the
// other two methods use, without needing any of spektrafilm's own
// per-sensitivity calibration constants.
static inline void _fe_hanatos_spectrum(const dt_iop_filmemulation_data_t *d,
                                        const float rgb[3],
                                        float spectrum[FE_N_WAVELENGTHS])
{
  float shape[FE_N_WAVELENGTHS];
  _fe_hanatos_reconstruct(rgb, shape);
  for(int i = 0; i < FE_N_WAVELENGTHS; i++)
    spectrum[i] = shape[i] / fmaxf(d->hanatos_ref_shape[i], 1e-6f);
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
// spektrafilm mode -- a from-scratch port of spektrafilm's own algorithm
// (see external/filmemulation_data_spektrafilm.c's own header for what was
// ported and why). Uses its own helper functions below rather than the
// "initial version" split-gauss cascade machinery above; the only things
// shared with it are _fe_tri2quad()/_fe_hanatos_fetch_coeffs() (both
// stock-independent physics of the shared Hanatos spectra basis) and
// _fe_finish_pixel() (gamut mapping + working-space conversion tail,
// identical for both data sets).
// ---------------------------------------------------------------------------

// Per-stock Hanatos-2025 spectral reconstruction (see _fe_hanatos_reconstruct()
// for the shared closed-form basis this specializes): the fixed Adobe-RGB-
// >XYZ matrix is replaced by this film's own CAT16-adapted matrix (Adobe
// RGB D65 white -> XYZ adapted to the film's reference illuminant -- see
// dt_sf_film_t.cat16_matrix's own comment), and the reconstruction is
// evaluated at spektrafilm's native 81-wavelength (380-780nm/5nm) grid
// instead of the "initial version"'s 31-point one. Also returns the tc
// (chromaticity-square) coordinate used, for the optional per-stock
// "surface" log-exposure correction below.
static inline void _sf_hanatos_reconstruct(const dt_sf_film_t *film, const float rgb_in[3],
                                            float spectrum[SF_N_WAVELENGTHS], float *tx_out, float *ty_out)
{
  const float r = fmaxf(rgb_in[0], 0.f), g = fmaxf(rgb_in[1], 0.f), b = fmaxf(rgb_in[2], 0.f);
  const float X = film->cat16_matrix[0][0] * r + film->cat16_matrix[0][1] * g + film->cat16_matrix[0][2] * b;
  const float Y = film->cat16_matrix[1][0] * r + film->cat16_matrix[1][1] * g + film->cat16_matrix[1][2] * b;
  const float Z = film->cat16_matrix[2][0] * r + film->cat16_matrix[2][1] * g + film->cat16_matrix[2][2] * b;
  const float sum = fmaxf(X + Y + Z, 1e-9f);

  float tx, ty;
  _fe_tri2quad(X / sum, Y / sum, &tx, &ty);
  *tx_out = tx;
  *ty_out = ty;

  float coeffs[4];
  _fe_hanatos_fetch_coeffs(tx, ty, coeffs);
  const float c3 = fmaxf(coeffs[3], 1e-6f);

  for(int i = 0; i < SF_N_WAVELENGTHS; i++)
  {
    const float wl = sf_wavelengths[i];
    const float x = (coeffs[0] * wl + coeffs[1]) * wl + coeffs[2];
    const float y = 1.f / sqrtf(x * x + 1.f);
    spectrum[i] = ((0.5f * x * y + 0.5f) / c3) * sum;
  }
}

// spektrafilm's poly2d_deg4() + hanika_sigmoid(): a degree-4 2D polynomial
// in tc-space, algebraically soft-clamped to +-2 stops -- see
// dt_sf_film_t.surface_poly's own comment.
static inline float _sf_eval_surface_poly(const float poly[14], const float dx, const float dy)
{
  const float x2 = dx * dx, y2 = dy * dy, xy = dx * dy;
  const float x3 = x2 * dx, y3 = y2 * dy;
  const float raw = poly[0] * dx + poly[1] * dy + poly[2] * x2 + poly[3] * y2 + poly[4] * xy
                   + poly[5] * x3 + poly[6] * y3 + poly[7] * (x2 * dy) + poly[8] * (dx * y2)
                   + poly[9] * (x2 * x2) + poly[10] * (y2 * y2) + poly[11] * (x3 * dy)
                   + poly[12] * (x2 * y2) + poly[13] * (dx * y3);
  const float max_val = 2.f;
  return raw / sqrtf(1.f + (raw / max_val) * (raw / max_val));
}

// RGB -> per-layer raw exposure: spectral reconstruction, dot with the
// film's (already Hanatos2025-window-adapted) sensitivity, optional
// per-stock "surface" correction -- matches spektrafilm's
// FilmingStage._rgb_to_film_raw() (the camera UV/IR band-pass filter, and
// the DIR-coupler-unrelated highlight-boost/diffusion-filter/halation
// steps of FilmingStage.expose(), are out of scope -- see this feature's
// commit message for why).
static inline void _sf_compute_raw_exposure(const dt_sf_film_t *film, const gboolean surface_active,
                                            const float rgb[3], float raw[3])
{
  float spectrum[SF_N_WAVELENGTHS];
  float tx, ty;
  _sf_hanatos_reconstruct(film, rgb, spectrum, &tx, &ty);

  for(int c = 0; c < 3; c++)
  {
    float e = 0.f;
    for(int w = 0; w < SF_N_WAVELENGTHS; w++) e += spectrum[w] * film->sensitivity[w][c];
    raw[c] = e;
  }

  if(surface_active)
  {
    const float dx = tx - film->tc_center[0];
    const float dy = ty - film->tc_center[1];
    for(int c = 0; c < 3; c++)
      raw[c] *= exp2f(_sf_eval_surface_poly(film->surface_poly[c], dx, dy));
  }
}

// Linear interpolation of a baked (log_exposure, density) curve against
// spektrafilm's shared 256-point log-exposure axis (sf_log_exposure) --
// clamps to the endpoint value outside range, matching spektrafilm's own
// fast_interp(). "curve" is any [SF_N_LOG_EXPOSURE][3] table (a film's own
// density_curves, or its DIR-coupler-precorrected variant).
static inline float _sf_interp_shared_axis(const float (*curve)[3], const int channel, const float x)
{
  const int n = SF_N_LOG_EXPOSURE;
  if(x <= sf_log_exposure[0]) return curve[0][channel];
  if(x >= sf_log_exposure[n - 1]) return curve[n - 1][channel];
  int lo = 0, hi = n - 1;
  while(hi - lo > 1)
  {
    const int mid = (lo + hi) / 2;
    if(sf_log_exposure[mid] <= x) lo = mid; else hi = mid;
  }
  const float t = (x - sf_log_exposure[lo]) / (sf_log_exposure[hi] - sf_log_exposure[lo]);
  return curve[lo][channel] * (1.f - t) + curve[hi][channel] * t;
}

// Linear interpolation against an arbitrary (not necessarily
// sf_log_exposure-aligned) ascending x-axis, one column of a
// [SF_N_LOG_EXPOSURE][3] pair of tables -- used only to precompute the
// DIR-coupler curve correction below (setup time, not per pixel).
static inline float _sf_interp_column(const float (*x_axis)[3], const float (*y_axis)[3], const int channel,
                                      const int n, const float x)
{
  if(x <= x_axis[0][channel]) return y_axis[0][channel];
  if(x >= x_axis[n - 1][channel]) return y_axis[n - 1][channel];
  int lo = 0, hi = n - 1;
  while(hi - lo > 1)
  {
    const int mid = (lo + hi) / 2;
    if(x_axis[mid][channel] <= x) lo = mid; else hi = mid;
  }
  const float t = (x - x_axis[lo][channel]) / (x_axis[hi][channel] - x_axis[lo][channel]);
  return y_axis[lo][channel] * (1.f - t) + y_axis[hi][channel] * t;
}

// Beer-Lambert spectral density compositing (film or paper, same formula):
// each dye channel's density weighted by its own spectral absorption,
// plus the base/fog spectral density -- matches spektrafilm's
// compute_density_spectral().
static inline void _sf_density_spectral(const float (*channel_density)[3], const float *base_density,
                                        const float density_cmy[3], float spectral[SF_N_WAVELENGTHS])
{
  for(int l = 0; l < SF_N_WAVELENGTHS; l++)
    spectral[l] = density_cmy[0] * channel_density[l][0] + density_cmy[1] * channel_density[l][1]
                + density_cmy[2] * channel_density[l][2] + base_density[l];
}

// Density -> transmitted light, zeroing wavelengths outside the profile's
// own measured spectral range (spectral_valid) instead of propagating the
// NaNs spektrafilm's own density_to_light() would produce there -- see
// dt_sf_film_t.spectral_valid's own comment.
static inline void _sf_density_to_light(const float *spectral, const float *illuminant, const float *valid,
                                        float light[SF_N_WAVELENGTHS])
{
  for(int l = 0; l < SF_N_WAVELENGTHS; l++)
    light[l] = (valid[l] > 0.5f) ? (powf(10.f, -spectral[l]) * illuminant[l]) : 0.f;
}

// spektrafilm's default enlarger dichroic color-head filter
// (color_filters.create_combined_dichroic_filter(), the "custom" closed-
// form erf filter set that color_enlarger() uses by default -- no measured
// brand CSV needed) applied to the TH-KG3 lamp spectrum at a given
// (c, m, y) Kodak-CC-unit filter setting.
static void _sf_enlarger_filtered_illuminant(const float cc[3], float out[SF_N_WAVELENGTHS])
{
  const float tc = powf(10.f, -cc[0] / 100.f);
  const float tm = powf(10.f, -cc[1] / 100.f);
  const float ty = powf(10.f, -cc[2] / 100.f);
  for(int l = 0; l < SF_N_WAVELENGTHS; l++)
  {
    const float wl = sf_wavelengths[l];
    const float y_pure = 0.5f * (1.f + erff((wl - 516.f) / 12.f));
    const float m_pure = (wl <= 550.f) ? 0.5f * (1.f - erff((wl - 500.f) / 8.f)) : 0.5f * (1.f + erff((wl - 610.f) / 8.f));
    const float c_pure = 0.5f * (1.f - erff((wl - 607.f) / 8.f));
    const float dim_c = 1.f - (1.f - c_pure) * (1.f - tc);
    const float dim_m = 1.f - (1.f - m_pure) * (1.f - tm);
    const float dim_y = 1.f - (1.f - y_pure) * (1.f - ty);
    out[l] = sf_enlarger_illuminant[l] * dim_c * dim_m * dim_y;
  }
}

// Film density -> print exposure: spectral density -> transmitted enlarger
// light -> paper's own spectral sensitivity -> exposure-factor-normalized
// raw -> log10 -- matches spektrafilm's PrintingStage._film_cmy_to_print_
// log_raw() (preflash is out of scope: off by default upstream too).
static inline void _sf_film_to_print_log_raw(const dt_sf_film_t *film, const dt_sf_paper_t *paper,
                                             const float enlarger_light[SF_N_WAVELENGTHS],
                                             const float exposure_factor, const float print_exposure,
                                             const float density_cmy_film[3], float log_raw_print[3])
{
  float spectral[SF_N_WAVELENGTHS];
  _sf_density_spectral(film->channel_density, film->base_density, density_cmy_film, spectral);
  float light[SF_N_WAVELENGTHS];
  _sf_density_to_light(spectral, enlarger_light, film->spectral_valid, light);

  float raw[3] = { 0.f, 0.f, 0.f };
  for(int l = 0; l < SF_N_WAVELENGTHS; l++)
    for(int c = 0; c < 3; c++)
      raw[c] += light[l] * paper->sensitivity[l][c];

  for(int c = 0; c < 3; c++)
  {
    raw[c] *= exposure_factor * print_exposure;
    log_raw_print[c] = log10f(fmaxf(raw[c], 0.f) + 1e-10f);
  }
}

// Paper's parametric print density-curve model ("cdfs": a sum of scaled
// normal CDFs per sub-layer) -- matches spektrafilm's utils.morph_curves
// _evaluate_channel_density()/apply_print_curves_morph(), specialized to
// a single uniform gamma_factor (this module's "print gamma" slider) in
// place of spektrafilm's 6 independent fast/slow/RGB gamma knobs and its
// optional developer-exhaustion Gumbel blend (out of scope: a creative
// refinement of an already-creative control, off by default upstream too).
// All bundled papers are profile type "negative" (sign = +1); see
// dt_sf_paper_t's own comment.
static inline void _sf_print_develop(const dt_sf_paper_t *paper, const float gamma,
                                     const float log_raw_print[3], float density_cmy_print[3])
{
  for(int c = 0; c < 3; c++)
  {
    float d = 0.f;
    for(int i = 0; i < 3; i++)
    {
      const float z = (gamma * log_raw_print[c] - paper->cdfs_centers[c][i]) / paper->cdfs_sigmas[c][i];
      d += paper->cdfs_amplitudes[c][i] * _fe_norm_cdf(z);
    }
    density_cmy_print[c] = d;
  }
}

// Density -> display RGB: spectral density -> transmitted viewing-
// illuminant light -> CIE XYZ -> linear Adobe RGB (D65) via a viewing-
// illuminant-adapted matrix (see sf_view_illuminant_to_adobergb's own
// comment) -- matches spektrafilm's ScanningStage.scan(). Takes the
// channel_density/base_density/spectral_valid trio directly (rather than a
// dt_sf_paper_t *) so the same function serves both the paper-print path
// (paper's own density response) and reversal/positive film viewed
// directly (film's own density response, no paper stage at all -- see
// _sf_process_simple()'s own comment on why). Glare and output gamut
// compression are out of scope: darktable's own bloom/hazeremoval and
// colorbalancergb modules cover similar ground; _fe_finish_pixel()'s
// hue-preserving gamut mapping handles the final [0,1] clamp instead of
// spektrafilm's own chroma/lightness compression.
static inline void _sf_scan_to_rgb(const float (*channel_density)[3], const float *base_density,
                                   const float *spectral_valid, const float *view_illuminant,
                                   const float view_norm, const float (*view_to_adobergb)[3],
                                   const float density_cmy[3], float rgb_adobergb[3])
{
  float spectral[SF_N_WAVELENGTHS];
  _sf_density_spectral(channel_density, base_density, density_cmy, spectral);
  float light[SF_N_WAVELENGTHS];
  _sf_density_to_light(spectral, view_illuminant, spectral_valid, light);

  float xyz[3] = { 0.f, 0.f, 0.f };
  for(int l = 0; l < SF_N_WAVELENGTHS; l++)
    for(int k = 0; k < 3; k++)
      xyz[k] += light[l] * sf_cmfs[l][k];
  const float inv_norm = 1.f / fmaxf(view_norm, 1e-9f);
  for(int k = 0; k < 3; k++) xyz[k] *= inv_norm;

  for(int c = 0; c < 3; c++)
    rgb_adobergb[c] = view_to_adobergb[c][0] * xyz[0] + view_to_adobergb[c][1] * xyz[1]
                    + view_to_adobergb[c][2] * xyz[2];
}

// Film density -> final display RGB: for positive (reversal/slide) film,
// scan the film's own density directly under its own viewing illuminant
// (see dt_iop_filmemulation_sf_data_t.film_view_illuminant's own comment
// for why); for negative film, run the full film->paper printing cascade
// and scan the resulting paper density instead.
static inline void _sf_film_density_to_rgb(const dt_sf_film_t *film, const dt_sf_paper_t *paper,
                                           const dt_iop_filmemulation_sf_data_t *sf,
                                           const float density_cmy_film[3], float rgb_adobergb[3])
{
  if(film->positive)
  {
    _sf_scan_to_rgb(film->channel_density, film->base_density, film->spectral_valid,
                    sf->film_view_illuminant, sf->film_view_norm, sf->film_view_to_adobergb,
                    density_cmy_film, rgb_adobergb);
    return;
  }

  float log_raw_print[3];
  _sf_film_to_print_log_raw(film, paper, sf->enlarger_light, sf->exposure_factor, sf->print_exposure,
                            density_cmy_film, log_raw_print);

  float density_cmy_print[3];
  _sf_print_develop(paper, sf->print_gamma, log_raw_print, density_cmy_print);

  _sf_scan_to_rgb(paper->channel_density, paper->base_density, paper->spectral_valid,
                  sf->view_illuminant, sf->view_norm, sf->view_to_adobergb,
                  density_cmy_print, rgb_adobergb);
}

// Setup-time-only (commit_params) print-exposure normalizer: exposes a
// spatially-uniform 18% grey scene patch through the film and enlarger,
// onto the paper's own sensitivity, and returns the reciprocal of the
// geometric-mean raw response -- so the print stage's absolute exposure
// tracks the film+paper pairing's own midgray reproduction regardless of
// the camera-exposure slider (spektrafilm's "print_exposure_compensation"
// + "normalize_print_exposure", both on by default) -- matches
// PrintingStage._compute_exposure_factor_midgray()/_exposure_factor().
static float _sf_compute_exposure_factor(const dt_sf_film_t *film, const dt_sf_paper_t *paper,
                                         const float enlarger_light[SF_N_WAVELENGTHS],
                                         const float camera_exposure_mul, const gboolean surface_active)
{
  const float grey = 0.184f * camera_exposure_mul;
  const float rgb_mid[3] = { grey, grey, grey };
  float raw[3];
  _sf_compute_raw_exposure(film, surface_active, rgb_mid, raw);

  float density_cmy[3];
  for(int c = 0; c < 3; c++)
    density_cmy[c] = _sf_interp_shared_axis(film->density_curves, c, log10f(raw[c] + 1e-10f));

  float spectral[SF_N_WAVELENGTHS];
  _sf_density_spectral(film->channel_density, film->base_density, density_cmy, spectral);
  float light[SF_N_WAVELENGTHS];
  _sf_density_to_light(spectral, enlarger_light, film->spectral_valid, light);

  float raw_mid[3] = { 0.f, 0.f, 0.f };
  for(int l = 0; l < SF_N_WAVELENGTHS; l++)
    for(int c = 0; c < 3; c++)
      raw_mid[c] += light[l] * paper->sensitivity[l][c];

  float logsum = 0.f;
  for(int c = 0; c < 3; c++) logsum += logf(fmaxf(raw_mid[c], 1e-10f));
  const float geomean = expf(logsum / 3.f);
  return 1.f / fmaxf(geomean, 1e-10f);
}

// Setup-time-only (commit_params): the density curves a film would need to
// have, before DIR-coupler self/cross-inhibition, to reproduce its own
// real (measured) density_curves after the correction is applied -- so a
// spatially-uniform field's gray-ramp reproduction is exactly invariant to
// whether DIR couplers are active. Matches spektrafilm's
// compute_density_curves_before_dir_couplers() (its positive/negative
// sign-flip branch is a no-op once negation is distributed through linear
// interpolation on both sides, so this implements just the one formula).
static void _sf_setup_dir_couplers_curves(const dt_sf_film_t *film, const float matrix[3][3],
                                          float density_curves_0[SF_N_LOG_EXPOSURE][3])
{
  const int n = SF_N_LOG_EXPOSURE;
  float log_exposure_0[SF_N_LOG_EXPOSURE][3];

  for(int i = 0; i < n; i++)
  {
    float silver[3];
    for(int c = 0; c < 3; c++)
      silver[c] = film->positive ? (film->dmax[c] - film->density_curves[i][c]) : film->density_curves[i][c];

    for(int recv = 0; recv < 3; recv++)
    {
      float acc = 0.f;
      for(int donor = 0; donor < 3; donor++) acc += silver[donor] * matrix[donor][recv];
      log_exposure_0[i][recv] = sf_log_exposure[i] - acc;
    }
  }

  for(int c = 0; c < 3; c++)
    for(int i = 0; i < n; i++)
      density_curves_0[i][c] = _sf_interp_column(log_exposure_0, film->density_curves, c, n, sf_log_exposure[i]);
}

// commit_params() dispatch target for dataset == DT_FE_DATASET_SPEKTRAFILM.
static void _sf_commit_params(const dt_iop_filmemulation_params_t *p, dt_iop_filmemulation_data_t *d)
{
  dt_iop_filmemulation_sf_data_t *sf = &d->sf;
  sf->active = TRUE;
  sf->film = &dt_sf_films[p->sf_film];
  sf->paper = &dt_sf_papers[p->sf_paper];
  sf->surface_active = p->sf_hanatos_surface;
  sf->camera_exposure_mul = exp2f(p->sf_camera_ev);
  sf->print_exposure = p->sf_print_exposure;
  sf->print_gamma = p->sf_print_gamma;

  sf->dir_couplers_active = p->dir_couplers_active;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      sf->dir_couplers_matrix[i][j] = sf->film->couplers_matrix[i][j] * p->dir_couplers_amount;
  if(sf->dir_couplers_active)
    _sf_setup_dir_couplers_curves(sf->film, sf->dir_couplers_matrix, sf->density_curves_0);

  const float *nf = sf_neutral_filters[(int)p->sf_paper][(int)p->sf_film];
  float cc[3];
  if(nf[0] >= 0.f) { cc[0] = nf[0]; cc[1] = nf[1]; cc[2] = nf[2]; }
  else { cc[0] = 0.f; cc[1] = 65.f; cc[2] = 55.f; } // EnlargerParams generic defaults
  _sf_enlarger_filtered_illuminant(cc, sf->enlarger_light);

  sf->exposure_factor = _sf_compute_exposure_factor(sf->film, sf->paper, sf->enlarger_light,
                                                    sf->camera_exposure_mul, sf->surface_active);

  sf->view_illuminant = sf_view_illuminants[sf->paper->viewing_illuminant];
  float norm = 0.f;
  for(int l = 0; l < SF_N_WAVELENGTHS; l++) norm += sf->view_illuminant[l] * sf_cmfs[l][1];
  sf->view_norm = norm;
  sf->view_to_adobergb = sf_view_illuminant_to_adobergb[sf->paper->viewing_illuminant];

  sf->film_view_illuminant = sf_view_illuminants[sf->film->viewing_illuminant];
  float film_norm = 0.f;
  for(int l = 0; l < SF_N_WAVELENGTHS; l++) film_norm += sf->film_view_illuminant[l] * sf_cmfs[l][1];
  sf->film_view_norm = film_norm;
  sf->film_view_to_adobergb = sf_view_illuminant_to_adobergb[sf->film->viewing_illuminant];
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

  d->sf.active = FALSE;
  if(p->dataset == DT_FE_DATASET_SPEKTRAFILM)
  {
    _sf_commit_params(p, d);
    return;
  }

  d->use_hk = p->use_hk;

  d->spectral_method = p->spectral_method;
  if(d->spectral_method == DT_FE_SPECTRAL_HANATOS2025)
  {
    const float white[3] = { 1.f, 1.f, 1.f };
    _fe_hanatos_reconstruct(white, d->hanatos_ref_shape);
  }

  if(p->type == DT_FE_TYPE_BW)
  {
    d->n_layers = 1;
    d->dir_couplers_active = FALSE; // no inter-layer crosstalk with a single layer
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
    _fe_configure_dir_couplers(d, p, &film->dir_couplers, FALSE);
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
    _fe_configure_dir_couplers(d, p, &film->dir_couplers, TRUE);
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
    _fe_configure_dir_couplers(d, p, &film->dir_couplers, TRUE);
  }
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_filmemulation_data_t));
}

// ---------------------------------------------------------------------------
// process helpers
// ---------------------------------------------------------------------------

// pipe RGB -> linear Adobe RGB primaries -> spectral reconstruction ->
// optional HK -> per-layer linear exposure. Shared by both process() paths
// below so the spectral-reconstruction code exists exactly once.
static inline void _fe_compute_layer_exposures(const dt_iop_filmemulation_data_t *d,
                                               const float *const restrict pix_in,
                                               const dt_colormatrix_t pipe_to_adobergb,
                                               float E[3])
{
  dt_aligned_pixel_t rgb;
  dt_apply_transposed_color_matrix(pix_in, pipe_to_adobergb, rgb);

  float spectrum[FE_N_WAVELENGTHS];
  switch(d->spectral_method)
  {
    case DT_FE_SPECTRAL_MALLETT2019:
      _fe_mallett_spectrum(rgb, spectrum);
      break;
    case DT_FE_SPECTRAL_HANATOS2025:
      _fe_hanatos_spectrum(d, rgb, spectrum);
      break;
    case DT_FE_SPECTRAL_JAKOB_HANIKA2019:
    default:
    {
      float coeffs[3];
      _fe_rgb_to_sigmoid_coeffs(rgb[0], rgb[1], rgb[2], coeffs);
      _fe_sigmoid_spectrum(coeffs, spectrum);
      break;
    }
  }

  const float hk = d->use_hk ? _fe_hk_mul(rgb[0], rgb[1], rgb[2]) : 1.f;

  for(int layer = 0; layer < d->n_layers; layer++)
  {
    float e = 0.f;
    for(int w = 0; w < FE_N_WAVELENGTHS; w++) e += spectrum[w] * d->layer[layer].weight[w];
    E[layer] = e * hk;
  }
}

// result (linear Adobe RGB reflectance, alpha already in result[3]) ->
// gamut-mapped pipe RGB. Shared tail end of both process() paths.
static inline void _fe_finish_pixel(dt_aligned_pixel_t result,
                                    const gboolean has_gamut_matrices,
                                    const dt_colormatrix_t yrg_input_matrix_trans,
                                    const dt_colormatrix_t yrg_output_matrix,
                                    const dt_colormatrix_t yrg_output_matrix_trans,
                                    const dt_colormatrix_t adobergb_to_pipe,
                                    float *const restrict pix_out)
{
  if(has_gamut_matrices)
    _fe_gamut_map(result, yrg_input_matrix_trans, yrg_output_matrix, yrg_output_matrix_trans);
  else
    for(int c = 0; c < 3; c++) result[c] = fminf(fmaxf(result[c], 0.f), 1.f);

  dt_apply_transposed_color_matrix(result, adobergb_to_pipe, pix_out);
}

// The original, single-pass per-pixel path (no DIR-coupler spatial
// diffusion): every pixel is independent, exactly as before task 02.
// Used whenever DIR couplers are inactive, and as the fallback if the
// buffered path below fails to allocate its scratch buffers.
static void _fe_process_simple(const dt_iop_filmemulation_data_t *d,
                               const float *const in, float *const out,
                               const size_t npixels,
                               const dt_colormatrix_t pipe_to_adobergb,
                               const dt_colormatrix_t adobergb_to_pipe,
                               const gboolean has_gamut_matrices,
                               const dt_colormatrix_t yrg_input_matrix_trans,
                               const dt_colormatrix_t yrg_output_matrix,
                               const dt_colormatrix_t yrg_output_matrix_trans)
{
  const int n_layers = d->n_layers;

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;

    float E[3] = { 0.f, 0.f, 0.f };
    _fe_compute_layer_exposures(d, pix_in, pipe_to_adobergb, E);

    dt_aligned_pixel_t result = { 0.f, 0.f, 0.f, 0.f };
    for(int layer = 0; layer < n_layers; layer++)
      result[layer] = _fe_xfer(&d->layer[layer], E[layer]);
    if(n_layers == 1)
    {
      result[1] = result[0];
      result[2] = result[0];
    }
    result[3] = pix_in[3];

    _fe_finish_pixel(result, has_gamut_matrices, yrg_input_matrix_trans, yrg_output_matrix,
                     yrg_output_matrix_trans, adobergb_to_pipe, pix_out);
  }
}

// DIR-coupler path: a first pixel pass computes, per pixel, the log-
// exposure position "lh" fed to stage 0 and that stage's first-pass
// density (used as the "silver density" the coupler matrix acts on); the
// resulting per-receiver-layer correction term is spatially diffused
// (Gaussian core + a second, wider Gaussian standing in for the
// exponential tail) exactly like real inhibitor diffusion in the emulsion;
// a second pixel pass subtracts the (blurred) correction from lh and
// re-runs the cascade from there. See task 02's task file for the full
// derivation. Returns FALSE (having freed everything) if scratch-buffer
// allocation fails, so the caller can fall back to _fe_process_simple().
static gboolean _fe_process_dir_couplers(const dt_iop_filmemulation_data_t *d,
                                         const float *const in, float *const out,
                                         const dt_iop_roi_t *const roi_in,
                                         const dt_colormatrix_t pipe_to_adobergb,
                                         const dt_colormatrix_t adobergb_to_pipe,
                                         const gboolean has_gamut_matrices,
                                         const dt_colormatrix_t yrg_input_matrix_trans,
                                         const dt_colormatrix_t yrg_output_matrix,
                                         const dt_colormatrix_t yrg_output_matrix_trans)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  float *const restrict lh_buf = dt_alloc_align_float(4 * npixels);
  float *const restrict correction_buf = dt_alloc_align_float(4 * npixels);
  float *const restrict blur_core_buf = dt_alloc_align_float(4 * npixels);
  float *const restrict blur_tail_buf = dt_alloc_align_float(4 * npixels);

  if(!lh_buf || !correction_buf || !blur_core_buf || !blur_tail_buf)
  {
    dt_free_align(lh_buf);
    dt_free_align(correction_buf);
    dt_free_align(blur_core_buf);
    dt_free_align(blur_tail_buf);
    return FALSE;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;

    float E[3] = { 0.f, 0.f, 0.f };
    _fe_compute_layer_exposures(d, pix_in, pipe_to_adobergb, E);

    float D0[3];
    for(int layer = 0; layer < 3; layer++)
    {
      const float lh = _fe_lh_from_E(&d->layer[layer], E[layer]);
      lh_buf[k + layer] = lh;
      D0[layer] = _fe_eval_stage(&d->layer[layer], 0, lh);
    }
    lh_buf[k + 3] = 0.f;

    float silver[3];
    for(int layer = 0; layer < 3; layer++)
      silver[layer] = d->dir_couplers_positive ? (d->dir_couplers_density_max[layer] - D0[layer]) : D0[layer];

    for(int receiver = 0; receiver < 3; receiver++)
    {
      float c = 0.f;
      for(int donor = 0; donor < 3; donor++) c += silver[donor] * d->dir_couplers_matrix[donor][receiver];
      correction_buf[k + receiver] = c;
    }
    correction_buf[k + 3] = 0.f;
  }

  // pixel_size_um assumes the image width maps to a 35mm-film-format
  // physical width, so the diffusion radius scales with output resolution
  // rather than being a fixed pixel count (same convention the halation
  // task uses).
  const float pixel_size_um = FE_FILM_WIDTH_UM / (float)roi_in->width;
  const float sigma_core_px = FE_DIRC_DIFFUSION_UM / pixel_size_um;
  const float sigma_tail_px = FE_DIRC_DIFFUSION_TAIL_UM / pixel_size_um;
  const dt_aligned_pixel_t blur_max = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
  const dt_aligned_pixel_t blur_min = { -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

  dt_gaussian_t *g_core
      = dt_gaussian_init(roi_in->width, roi_in->height, 4, blur_max, blur_min, sigma_core_px, DT_IOP_GAUSSIAN_ZERO);
  dt_gaussian_t *g_tail
      = dt_gaussian_init(roi_in->width, roi_in->height, 4, blur_max, blur_min, sigma_tail_px, DT_IOP_GAUSSIAN_ZERO);
  if(g_core && g_tail)
  {
    dt_gaussian_blur_4c(g_core, correction_buf, blur_core_buf);
    dt_gaussian_blur_4c(g_tail, correction_buf, blur_tail_buf);
  }
  else
  {
    // Extremely unlikely allocation failure inside dt_gaussian_init();
    // fall back to the unblurred (pointwise-only) correction rather than
    // crash or leave the buffers uninitialized.
    memcpy(blur_core_buf, correction_buf, sizeof(float) * 4 * npixels);
    memcpy(blur_tail_buf, correction_buf, sizeof(float) * 4 * npixels);
  }
  if(g_core) dt_gaussian_free(g_core);
  if(g_tail) dt_gaussian_free(g_tail);

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;

    dt_aligned_pixel_t result = { 0.f, 0.f, 0.f, 0.f };
    for(int layer = 0; layer < 3; layer++)
    {
      const float correction = (1.f - FE_DIRC_DIFFUSION_TAIL_WEIGHT) * blur_core_buf[k + layer]
                              + FE_DIRC_DIFFUSION_TAIL_WEIGHT * blur_tail_buf[k + layer];
      result[layer] = _fe_xfer_from_lh(&d->layer[layer], lh_buf[k + layer] - correction);
    }
    result[3] = pix_in[3];

    _fe_finish_pixel(result, has_gamut_matrices, yrg_input_matrix_trans, yrg_output_matrix,
                     yrg_output_matrix_trans, adobergb_to_pipe, pix_out);
  }

  dt_free_align(lh_buf);
  dt_free_align(correction_buf);
  dt_free_align(blur_core_buf);
  dt_free_align(blur_tail_buf);
  return TRUE;
}

// spektrafilm mode's per-pixel path (no DIR-coupler spatial diffusion):
// pipe RGB -> Adobe RGB -> per-stock spectral reconstruction -> film
// exposure/density -> film->paper printing cascade -> paper density-curve
// model -> viewing-illuminant scan -> Adobe RGB -> gamut-mapped pipe RGB.
static void _sf_process_simple(const dt_iop_filmemulation_data_t *d,
                               const float *const in, float *const out,
                               const size_t npixels,
                               const dt_colormatrix_t pipe_to_adobergb,
                               const dt_colormatrix_t adobergb_to_pipe,
                               const gboolean has_gamut_matrices,
                               const dt_colormatrix_t yrg_input_matrix_trans,
                               const dt_colormatrix_t yrg_output_matrix,
                               const dt_colormatrix_t yrg_output_matrix_trans)
{
  const dt_iop_filmemulation_sf_data_t *sf = &d->sf;
  const dt_sf_film_t *film = sf->film;
  const dt_sf_paper_t *paper = sf->paper;

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;

    dt_aligned_pixel_t rgb;
    dt_apply_transposed_color_matrix(pix_in, pipe_to_adobergb, rgb);

    float raw[3];
    _sf_compute_raw_exposure(film, sf->surface_active, rgb, raw);
    for(int c = 0; c < 3; c++) raw[c] *= sf->camera_exposure_mul;

    float log_raw[3];
    for(int c = 0; c < 3; c++) log_raw[c] = log10f(fmaxf(raw[c], 0.f) + 1e-10f);

    float density_cmy_film[3];
    for(int c = 0; c < 3; c++) density_cmy_film[c] = _sf_interp_shared_axis(film->density_curves, c, log_raw[c]);

    dt_aligned_pixel_t result = { 0.f, 0.f, 0.f, 0.f };
    _sf_film_density_to_rgb(film, paper, sf, density_cmy_film, result);
    result[3] = pix_in[3];

    _fe_finish_pixel(result, has_gamut_matrices, yrg_input_matrix_trans, yrg_output_matrix,
                     yrg_output_matrix_trans, adobergb_to_pipe, pix_out);
  }
}

// spektrafilm mode's DIR-coupler path -- same two-pass/spatial-diffusion
// structure as _fe_process_dir_couplers() (first pass: uncorrected film
// density as the "silver" proxy, matrix-multiply into a per-receiver
// correction, spatially diffuse it; second pass: re-develop from the
// corrected log-exposure against the DIR-coupler-precorrected curves),
// then the same film->paper->scan cascade as _sf_process_simple() above.
// Returns FALSE (having freed everything) if scratch-buffer allocation
// fails, so the caller can fall back to _sf_process_simple().
static gboolean _sf_process_dir_couplers(const dt_iop_filmemulation_data_t *d,
                                         const float *const in, float *const out,
                                         const dt_iop_roi_t *const roi_in,
                                         const dt_colormatrix_t pipe_to_adobergb,
                                         const dt_colormatrix_t adobergb_to_pipe,
                                         const gboolean has_gamut_matrices,
                                         const dt_colormatrix_t yrg_input_matrix_trans,
                                         const dt_colormatrix_t yrg_output_matrix,
                                         const dt_colormatrix_t yrg_output_matrix_trans)
{
  const dt_iop_filmemulation_sf_data_t *sf = &d->sf;
  const dt_sf_film_t *film = sf->film;
  const dt_sf_paper_t *paper = sf->paper;
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  float *const restrict log_raw_buf = dt_alloc_align_float(4 * npixels);
  float *const restrict correction_buf = dt_alloc_align_float(4 * npixels);
  float *const restrict blur_core_buf = dt_alloc_align_float(4 * npixels);
  float *const restrict blur_tail_buf = dt_alloc_align_float(4 * npixels);

  if(!log_raw_buf || !correction_buf || !blur_core_buf || !blur_tail_buf)
  {
    dt_free_align(log_raw_buf);
    dt_free_align(correction_buf);
    dt_free_align(blur_core_buf);
    dt_free_align(blur_tail_buf);
    return FALSE;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;

    dt_aligned_pixel_t rgb;
    dt_apply_transposed_color_matrix(pix_in, pipe_to_adobergb, rgb);

    float raw[3];
    _sf_compute_raw_exposure(film, sf->surface_active, rgb, raw);
    for(int c = 0; c < 3; c++) raw[c] *= sf->camera_exposure_mul;

    float log_raw[3];
    for(int c = 0; c < 3; c++) log_raw[c] = log10f(fmaxf(raw[c], 0.f) + 1e-10f);
    log_raw_buf[k + 0] = log_raw[0];
    log_raw_buf[k + 1] = log_raw[1];
    log_raw_buf[k + 2] = log_raw[2];
    log_raw_buf[k + 3] = 0.f;

    float D0[3], silver[3];
    for(int c = 0; c < 3; c++)
    {
      D0[c] = _sf_interp_shared_axis(film->density_curves, c, log_raw[c]);
      silver[c] = film->positive ? (film->dmax[c] - D0[c]) : D0[c];
    }

    for(int recv = 0; recv < 3; recv++)
    {
      float acc = 0.f;
      for(int donor = 0; donor < 3; donor++) acc += silver[donor] * sf->dir_couplers_matrix[donor][recv];
      correction_buf[k + recv] = acc;
    }
    correction_buf[k + 3] = 0.f;
  }

  const float pixel_size_um = FE_FILM_WIDTH_UM / (float)roi_in->width;
  const float sigma_core_px = FE_DIRC_DIFFUSION_UM / pixel_size_um;
  const float sigma_tail_px = FE_DIRC_DIFFUSION_TAIL_UM / pixel_size_um;
  const dt_aligned_pixel_t blur_max = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
  const dt_aligned_pixel_t blur_min = { -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

  dt_gaussian_t *g_core
      = dt_gaussian_init(roi_in->width, roi_in->height, 4, blur_max, blur_min, sigma_core_px, DT_IOP_GAUSSIAN_ZERO);
  dt_gaussian_t *g_tail
      = dt_gaussian_init(roi_in->width, roi_in->height, 4, blur_max, blur_min, sigma_tail_px, DT_IOP_GAUSSIAN_ZERO);
  if(g_core && g_tail)
  {
    dt_gaussian_blur_4c(g_core, correction_buf, blur_core_buf);
    dt_gaussian_blur_4c(g_tail, correction_buf, blur_tail_buf);
  }
  else
  {
    memcpy(blur_core_buf, correction_buf, sizeof(float) * 4 * npixels);
    memcpy(blur_tail_buf, correction_buf, sizeof(float) * 4 * npixels);
  }
  if(g_core) dt_gaussian_free(g_core);
  if(g_tail) dt_gaussian_free(g_tail);

  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;

    float log_raw_corrected[3];
    for(int c = 0; c < 3; c++)
    {
      const float correction = (1.f - FE_DIRC_DIFFUSION_TAIL_WEIGHT) * blur_core_buf[k + c]
                              + FE_DIRC_DIFFUSION_TAIL_WEIGHT * blur_tail_buf[k + c];
      log_raw_corrected[c] = log_raw_buf[k + c] - correction;
    }

    float density_cmy_film[3];
    for(int c = 0; c < 3; c++)
      density_cmy_film[c] = _sf_interp_shared_axis(sf->density_curves_0, c, log_raw_corrected[c]);

    dt_aligned_pixel_t result = { 0.f, 0.f, 0.f, 0.f };
    _sf_film_density_to_rgb(film, paper, sf, density_cmy_film, result);
    result[3] = pix_in[3];

    _fe_finish_pixel(result, has_gamut_matrices, yrg_input_matrix_trans, yrg_output_matrix,
                     yrg_output_matrix_trans, adobergb_to_pipe, pix_out);
  }

  dt_free_align(log_raw_buf);
  dt_free_align(correction_buf);
  dt_free_align(blur_core_buf);
  dt_free_align(blur_tail_buf);
  return TRUE;
}

// ---------------------------------------------------------------------------
// process: pipe RGB -> linear Adobe RGB primaries -> spectral reconstruction
// -> exposure integration per layer -> optional HK -> optional DIR-coupler
// spatial diffusion -> cascade -> hue-preserving gamut mapping
// (_fe_gamut_map()) -> linear Adobe RGB primaries -> pipe RGB.
// (dataset == DT_FE_DATASET_SPEKTRAFILM dispatches to _sf_process_simple()/
// _sf_process_dir_couplers() instead -- see dt_iop_filmemulation_sf_data_t.)
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

  if(d->sf.active)
  {
    if(!d->sf.dir_couplers_active
       || !_sf_process_dir_couplers(d, in, out, roi_in, pipe_to_adobergb, adobergb_to_pipe, has_gamut_matrices,
                                    yrg_input_matrix_trans, yrg_output_matrix, yrg_output_matrix_trans))
    {
      _sf_process_simple(d, in, out, npixels, pipe_to_adobergb, adobergb_to_pipe, has_gamut_matrices,
                         yrg_input_matrix_trans, yrg_output_matrix, yrg_output_matrix_trans);
    }
    return;
  }

  const gboolean use_dir_couplers = d->dir_couplers_active && d->n_layers == 3;

  if(!use_dir_couplers
     || !_fe_process_dir_couplers(d, in, out, roi_in, pipe_to_adobergb, adobergb_to_pipe, has_gamut_matrices,
                                  yrg_input_matrix_trans, yrg_output_matrix, yrg_output_matrix_trans))
  {
    _fe_process_simple(d, in, out, npixels, pipe_to_adobergb, adobergb_to_pipe, has_gamut_matrices,
                       yrg_input_matrix_trans, yrg_output_matrix, yrg_output_matrix_trans);
  }
}

// Marks the selected film's own real-world companion paper (dt_sf_film_t.
// default_paper, from spektrafilm's target_print) with a trailing "*" in
// the paper combobox -- darktable has no standard "recommended entry"
// convention (dt_bauhaus_combobox_set_entry_label(), used elsewhere for
// denoiseprofile.c's auto-profile name, is the closest building block).
static void _sf_mark_companion_paper(const dt_iop_filmemulation_gui_data_t *g, const int film_idx)
{
  const int default_paper = dt_sf_films[film_idx].default_paper;
  for(int i = 0; i < SF_N_PAPERS; i++)
  {
    char label[128];
    if(i == default_paper)
      g_snprintf(label, sizeof(label), "%s *", dt_sf_papers[i].display_name);
    else
      g_strlcpy(label, dt_sf_papers[i].display_name, sizeof(label));
    dt_bauhaus_combobox_set_entry_label(g->sf_paper, i, label);
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

  if(!w || w == g->dataset || w == g->sf_film)
    _sf_mark_companion_paper(g, (int)p->sf_film);

  if(!w || w == g->dataset)
  {
    gtk_widget_set_visible(g->box_initial, p->dataset == DT_FE_DATASET_INITIAL);
    gtk_widget_set_visible(g->box_spektrafilm, p->dataset == DT_FE_DATASET_SPEKTRAFILM);
  }
  if(!w || w == g->dataset || w == g->type)
  {
    // dye-coupler crosstalk is shared by both data sets, but only makes
    // sense with 3 emulsion layers -- the initial-version data set's B&W
    // stock has just one.
    const gboolean show_dir_couplers = (p->dataset == DT_FE_DATASET_SPEKTRAFILM)
                                      || (p->dataset == DT_FE_DATASET_INITIAL && p->type == DT_FE_TYPE_COLOR);
    gtk_widget_set_visible(g->dir_couplers_active, show_dir_couplers);
    gtk_widget_set_visible(g->dir_couplers_amount, show_dir_couplers);
  }
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

  g->dataset = dt_bauhaus_combobox_from_params(self, "dataset");
  gtk_widget_set_tooltip_text(g->dataset,
                              _("initial version: darktable's own film/paper data and cascade.\n"
                                "spektrafilm: a straight port of the spektrafilm project's own film/\n"
                                "paper profiles and rendering algorithm (per-stock spectral-sensitivity\n"
                                "calibration, a real film->paper spectral printing cascade through an\n"
                                "enlarger, and its own parametric print response model)."));
  // self->widget is lazily created (as a vbox) by the first
  // dt_bauhaus_*_from_params() call above -- capture it only now.
  GtkWidget *main_box = self->widget;

  g->box_initial = self->widget = dt_gui_vbox();

  g->type = dt_bauhaus_combobox_from_params(self, "type");

  g->box_bw = self->widget = dt_gui_vbox();
  g->bw_film = dt_bauhaus_combobox_from_params(self, "bw_film");
  g->bw_filter = dt_bauhaus_combobox_from_params(self, "bw_filter");
  gtk_widget_set_tooltip_text(g->bw_filter, _("Wratten glass filter -- folded into the film's spectral response"));
  dt_gui_box_add(g->box_initial, g->box_bw);

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
  dt_gui_box_add(g->box_initial, g->box_color);

  self->widget = g->box_initial;

  g->spectral_method = dt_bauhaus_combobox_from_params(self, "spectral_method");
  gtk_widget_set_tooltip_text(g->spectral_method,
                              _("how a scene color is turned into the full-spectrum light the film's\n"
                                "layers actually react to. RGB never carries enough information to\n"
                                "know the real light spectrum, so this is always an educated guess;\n"
                                "the three methods mostly agree for ordinary daylight/tungsten photos\n"
                                "and can diverge for narrow-band or heavily colored light (colored\n"
                                "LEDs, sodium-vapor, neon, stage lighting)."));

  g->print_gamma = dt_bauhaus_slider_from_params(self, "print_gamma");
  gtk_widget_set_tooltip_text(g->print_gamma, _("Jones system-gamma target: how contrasty the film+paper pairing\n"
                                          "renders overall (\"punchiness\"). 1.0 is colorimetrically faithful;\n"
                                          "real print materials are usually viewed punchier than that."));
  g->use_hk = dt_bauhaus_toggle_from_params(self, "use_hk");
  gtk_widget_set_tooltip_text(g->use_hk, _("account for saturated colors appearing brighter than their\n"
                                           "measured luminance (Helmholtz-Kohlrausch effect)"));

  dt_gui_box_add(main_box, g->box_initial);

  g->box_spektrafilm = self->widget = dt_gui_vbox();
  g->sf_film = dt_bauhaus_combobox_from_params(self, "sf_film");
  g->sf_paper = dt_bauhaus_combobox_from_params(self, "sf_paper");
  gtk_widget_set_tooltip_text(g->sf_paper, _("the paper the selected film was actually calibrated/printed\n"
                                          "against is marked with a trailing *, but any pairing works"));
  g->sf_camera_ev = dt_bauhaus_slider_from_params(self, "sf_camera_ev");
  gtk_widget_set_tooltip_text(g->sf_camera_ev, _("camera exposure compensation, in stops -- the print stage\n"
                                          "auto-compensates so the film+paper pairing's own midgray stays put"));
  g->sf_print_exposure = dt_bauhaus_slider_from_params(self, "sf_print_exposure");
  gtk_widget_set_tooltip_text(g->sf_print_exposure, _("enlarger exposure onto the paper"));
  g->sf_print_gamma = dt_bauhaus_slider_from_params(self, "sf_print_gamma");
  gtk_widget_set_tooltip_text(g->sf_print_gamma, _("print contrast (\"punchiness\"): rescales the paper's own\n"
                                          "fitted density-curve model. 1.0 matches the paper's real response."));
  g->sf_hanatos_surface = dt_bauhaus_toggle_from_params(self, "sf_hanatos_surface");
  gtk_widget_set_tooltip_text(g->sf_hanatos_surface,
                              _("apply this film's own fitted correction to the shared spectral-\n"
                                "upsampling basis (spektrafilm's Hanatos2025 \"surface\" adaptation,\n"
                                "off by default upstream too) -- a small additional per-chromaticity\n"
                                "refinement, most visible on saturated colors"));
  dt_gui_box_add(main_box, g->box_spektrafilm);

  self->widget = main_box;

  g->dir_couplers_active = dt_bauhaus_toggle_from_params(self, "dir_couplers_active");
  gtk_widget_set_tooltip_text(g->dir_couplers_active,
                              _("simulate DIR (development-inhibitor-releasing) coupler crosstalk\n"
                                "between the R/G/B emulsion layers -- the real film chemistry behind\n"
                                "color negative/reversal film's characteristic saturation and\n"
                                "local-contrast response"));
  g->dir_couplers_amount = dt_bauhaus_slider_from_params(self, "dir_couplers_amount");
  gtk_widget_set_tooltip_text(g->dir_couplers_amount, _("strength of the dye coupler interaction"));
}

// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-space on;
