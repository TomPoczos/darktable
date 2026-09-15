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

/*
DOCUMENTATION
This module implements a scene-referred local contrast enhancement algorithm,
designed to enhance local details while preserving edges and avoiding artifacts.

It builds upon the original proof-of-concept algorithm proposed by WileCoyote:
https://discuss.pixls.us/t/experiments-with-a-scene-referred-local-contrast-module-proof-of-concept/55402

And then further explored and optimized by Christian Bouhon
https://discuss.pixls.us/t/contrast-management-rgb-a-new-scene-referred-approach-poc/56004

Current status as implemented by Jandren:
- Local contrast in log space based on the eigf surface blur filter.
*/

/* Work around a compiler bug, see
   https://github.com/darktable-org/darktable/issues/21801
*/
#if __GNUC__ == 16
#pragma GCC optimize ("no-ipa-cp")
#endif

#include "common/extra_optimizations.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/fast_guided_filter.h"
#include "common/color_picker.h"
#include "common/eigf.h"
#include "common/gaussian.h"
#include "common/luminance_mask.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/preview_data.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "dtgtk/paint.h"
#include "dtgtk/togglebutton.h"
#include "dtgtk/expander.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/choleski.h"
#include "iop/iop_api.h"
#include "libs/lib.h"
#include "common/iop_group.h"

#ifdef _OPENMP
#include <omp.h>
#endif

DT_MODULE_INTROSPECTION(4, dt_iop_contrast_params_t)

#define CT_BANDS 9          // detail levels 2..10, one node per octave
#define CT_BAND_D0 2.0f     // detail level of the coarsest node

// §3.3/research.md §2.4, re-anchored by implementation-plan-4.md §2.1:
// eps_k = eps * 2^(p * (CT_BANDS-1-k)), k the band's own nominal ladder
// index (finest = CT_BANDS-1) -- a fixed property of the band, not of which
// band happened to survive modify_roi_in at the current roi scale. p is
// deliberately small and un-exposed ("a small p, not another slider") --
// 0.3 is a plausible starting point sized against phase0-band-energy.md's
// own numbers (raising the *global* eps from 0.2 to 0.8, a 4x change, was
// enough to keep every band alive there), not a value re-validated with
// phase0's own visual A/B method against real images; revisit if a future
// pass finds coarse bands still collapsing, or overshooting into halos, at
// this setting.
#define CT_FEATHERING_EXPONENT 0.3f

// the graph: nodes run coarse (left) to fine (right), one per octave, so the
// x axis is simply k/CT_BANDS (implementation-plan-3.md §4.2 replaces this
// with the projection grid's own span -- see _spectrum_lambda_to_raw_x).
//
// implementation-plan-3.md §4.1: the y axis is log2 gain, symmetric about
// the neutral 1.0, half-range log2(5.0) -- so the axis runs 0.2 .. 5.0 and
// is exactly the band parameter's own hard range ($MAX 5.0 and its
// reciprocal). The old linear CT_GRAPH_Y_MAX = 2.0 could not draw
// CT_EQUALIZE_GAIN_HI at all (2.5 clamped to the top edge, pixel-identical
// to 2.0) and split the envelope 35%/50% of the height between its
// 1.737-octave cut half and its 1.0-octave boost half, on a quantity where
// a factor is a factor either way.
#define CT_GRAPH_LOG_HALF 2.3219281   // log2(5.0)
#define CT_GRAPH_RES 64      // curve points sampled between nodes, per implementation-plan.md §1.4

// the detail-scale ladder (picked-region measurement below, and the §2.1
// frame-wide one), shared by both: DoG bands per octave, the finest sigma,
// and how many octaves/bands the ladder is allowed to grow to.
#define CT_SCALES_PER_OCTAVE 3
#define CT_SIGMA_BASE 1.2f
#define CT_MAX_OCTAVES 12
#define CT_MAX_BANDS (CT_MAX_OCTAVES * CT_SCALES_PER_OCTAVE)

// implementation-plan-3.md §1.1: dt_gaussian_blur is Deriche's second-order
// recursive smoother with alpha = 1.695/sigma (src/common/gaussian.c,
// _compute_gauss_params). The operator it approximates,
// k*(alpha|x| + 1)*exp(-alpha|x|), has variance 4/alpha^2 -- so its standard
// deviation is 2/alpha = 1.1799*sigma, not sigma: 1.695 is Deriche's constant
// for the best *shape* match to a Gaussian, which is not the same criterion
// as matching its second moment, and gaussian.c does not say which one it
// wanted. Measured on the shipped recursion itself (delta impulse through
// exactly its forward/backward passes, picker-regression/harness_v2/
// dig_gaussian_sigma.py): 1.1799 for sigma >= 1.5, 1.1763 at sigma = 1.2,
// 1.169 at 0.92 -- uniform to 0.4% over every sigma this ladder uses, with DC
// gain exactly 1 and mean exactly 0.
//
// The ladder's rung *ratios* are unaffected (the factor is constant, and
// §2's incremental construction and its decimation step both carry it through
// unchanged); only the absolute label is, by 0.236 octave, always toward
// fine. Correct the label rather than pre-dividing the requested sigma: the
// factor is not constant below sigma ~= 0.8, so pre-dividing would need a
// per-sigma inversion and would still land the finest rung on the part of the
// curve where it is least uniform. Not applied to sigmas that have no
// dt_gaussian_blur behind them -- see _ct_band_sigma.
#define CT_GAUSSIAN_SIGMA_FACTOR 1.1799

// implementation-plan-2.md §3.1: peak wavelength of the DoG between sigma and
// 2^(1/CT_SCALES_PER_OCTAVE)*sigma -- lambda = pi*sqrt(2*(k2-1)/ln k2) * sigma,
// k2 = 2^(2/CT_SCALES_PER_OCTAVE) -- _band_peak_lambda's formula specialised
// to the ladder's own fixed rung ratio. The single conversion factor between
// a rung's sigma and the wavelength label it is published under; recompute if
// CT_SCALES_PER_OCTAVE ever changes.
#define CT_SIGMA_TO_LAMBDA 5.0091626

// §2.1: the frame-wide DoG ladder's block energy tables. Declared here,
// ahead of its own section further down, because dt_iop_contrast_gui_data_t
// needs the type; see that section for what builds and frees one.
typedef struct _ct_ladder_t
{
  int    nrungs;
  double sigma[CT_MAX_BANDS];   // §3.1: rung's own lower-boundary sigma, level-0 pixels
  double lambda[CT_MAX_BANDS];  // band-centre wavelength (sigma * CT_SIGMA_TO_LAMBDA), level-0 pixels
  size_t bw, bh;                 // block grid, the same for every rung
  double *sat2;                  // Sum(b^2) over blocks, nrungs * (bw+1) * (bh+1) doubles
  double *sat1;                  // Sum(|b|), same layout
  double *sat_n;                  // implementation-plan-4.md §8.2: real level-pixel count per block, same layout -- the box-query denominator _measure_box/_spectrum_frame_wide read instead of assuming every block full
  // implementation-plan-8.md §5.4: raw (non-cumulative) per-block RMS, one
  // value per rung, aggregated to super-blocks of >= CT_SUPERBLOCK_MIN_PIX
  // level pixels (§5.2's noise-floor argument for why 64). Same nrungs *
  // (bw+1)*(bh+1) allocation as sat2/sat1/sat_n purely so every rung's
  // component sits at the same node stride in g->pd -- this is NOT a SAT,
  // node (y, x) for y < bh, x < bw holds block (by=y, bx=x)'s own aggregated
  // value directly (a super-block's footprint repeats its one value across
  // every base block it covers), and the padding row (y==bh) / column
  // (x==bw) the real SATs need for their zero border is left at 0 here too,
  // simply unused.
  double *blockrms;
  double noise_floor[CT_MAX_BANDS];  // §2.4: per-rung, frame-wide block-minimum noise estimate
} _ct_ladder_t;

typedef struct dt_iop_contrast_params_t
{
  float gain_local_contrast;  // $MIN: 0.0 $MAX: 20.0 $DEFAULT: 1.0  $DESCRIPTION: "gain"
  float band[CT_BANDS];       // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0
  float scale_shift;          // $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0 $DESCRIPTION: "node placement"
  float edge_protection;      // $MIN: -10.0 $MAX: 10.0 $DEFAULT: 0.0 $DESCRIPTION: "adjust edge protection"
  int filter_iterations;      // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter iterations"
  float noise_bias;           // $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.001 $DESCRIPTION: "noise bias"
} dt_iop_contrast_params_t;

typedef struct dt_iop_contrast_data_t
{
  float gain_local_contrast;
  float scale_shift;
  float band[CT_BANDS];        // gains, copied verbatim from params
  int   nbands;                 // bands that survive the current roi scale
  float sigma[CT_BANDS];        // boundary sigmas, finest-surviving-first, in pixels of this roi
  float gain[CT_BANDS];         // matching gains, same order as sigma
  float feathering[CT_BANDS];   // §3.3: per-band edge protection, same order as sigma/gain
  float feathering_base;        // from edge_protection/filter_iterations, before the per-band scaling
  // implementation-plan-6.md §5.3/§6 Phase 6, revised implementation-plan-7.md
  // §4.1(d)/§6 Phase 1.3: gain_local_contrast after the per-band countershading
  // knee, one value per band (same order as sigma/gain), computed once
  // modify_roi_in knows d->sigma[]/d->gain[] -- process() reads this, not
  // gain_local_contrast directly.
  float band_master[CT_BANDS];
  int iterations;
  float noise_bias;
} dt_iop_contrast_data_t;

// values 0 .. CT_BANDS-1 select one band's own raw b_k (param-space index,
// coarsest = 0); the two named values above that select an accumulated view
// instead (implementation-plan.md §1.6).
typedef enum dt_iop_details_display_t
{
  DT_CT_MASK_OFF = -1,
  DT_CT_MASK_CORRECTION = CT_BANDS,      // sum (g_k - 1) b_k, i.e. what the module is doing
  DT_CT_MASK_DETAIL     = CT_BANDS + 1   // sum b_k, the un-gained detail texture
} dt_iop_details_display_t;

typedef struct dt_iop_contrast_gui_data_t
{
  // Flags
  dt_iop_details_display_t details_display;

  // GTK widgets
  GtkWidget *gain_local_contrast;
  GtkWidget *band[CT_BANDS];
  GtkWidget *scale_shift;
  GtkWidget *edge_protection;
  GtkWidget *filter_iterations;
  GtkWidget *noise_bias;

  // the graph (implementation-plan.md §1.4): a drawing area showing the nine
  // nodes as a curve, and a GtkStack toggled by middle-click on the graph
  // between that graph and the plain slider list.
  GtkDrawingArea *area;
  GtkStack *stack;
  dt_draw_curve_t *curve;

  // which bands survive at the current pipe scale, published from process()
  // under dt_iop_gui_enter/leave_critical_section (§1.5). nbands defaults to
  // CT_BANDS -- everything resolvable -- until the first preview pass lands.
  int nbands;

  // rms-based divisor of the currently displayed texture mask (a band or
  // DETAIL, never CORRECTION), published from process() the same way (§1.6)
  float mask_divisor;

  // graph interaction state
  gboolean dragging;
  int drag_band;   // node being dragged, -1 if none
  int hover_band;  // node nearest the pointer, -1 if none or not hovering

  // §2.2: the frame-wide DoG ladder's block SAT tables, republished through
  // dt_preview_data_t each untiled preview pass while the module is
  // expanded (§2.1's _build_ladder does the actual building). pd's buffer
  // is laid out node-major: (bw+1) x (bh+1) SAT nodes, 4*nrungs floats per
  // node (Sum(b^2), Sum(|b|), block pixel count and the non-cumulative
  // block rms interleaved per rung -- implementation-plan-4.md §8.2 and
  // implementation-plan-8.md §5.4, see _ladder_fill_cb) -- pd.width/height
  // are therefore the SAT dimensions, one more than the block grid on each
  // axis. ladder_nrungs/
  // lambda are the ladder metadata dt_preview_data_t has no room for;
  // protected by the same self->gui_lock dt_preview_data_t itself uses
  // (dt_iop_gui_enter/leave_critical_section), since pd.module == self.
  dt_preview_data_t pd;
  int ladder_nrungs;
  double ladder_sigma[CT_MAX_BANDS];   // §3.1: rung's own lower-boundary sigma, level-0 px
  double ladder_lambda[CT_MAX_BANDS];
  double ladder_noise_floor[CT_MAX_BANDS];  // §2.4, frame-wide, published the same way
  dt_iop_roi_t ladder_roi_in;  // the roi_in the ladder above was built from

  // §3.1: per-band block SAT tables for the module's own delivered bands,
  // built alongside the ladder in the same guarded preview pass (see
  // _decompose_and_accumulate's optional _ct_band_tables_t argument) and
  // queried the same way to calibrate the picker's linear H_k model against
  // what eigf's edge-awareness actually delivers (research.md §5.8).
  // components is fixed at 2*CT_BANDS + 1 (Sum(b^2)/Sum(|b|) per band plus
  // one shared block pixel count, see _band_fill_cb) -- unlike the ladder's
  // nrungs, CT_BANDS never changes, so this pd needs no resize-on-change dance.
  dt_preview_data_t band_pd;
  int band_nbands;               // how many of CT_BANDS survived the pass that built band_pd
  float band_sigma[CT_BANDS];    // finest-first, pixels of that same pass's roi (== ladder_roi_in)

  // §3.2: the last successful pick's own raw per-rung spectrum and fit, for
  // the graph's live overlay -- a record of the last measurement, drawn
  // every _area_draw regardless of whether the picker itself is still
  // "fresh". The fit's scalar fields are stored individually rather than as
  // a _ct_fit_t so this struct doesn't need that type's (later) definition.
  gboolean spectrum_valid;
  int spectrum_nrungs;
  double spectrum_lambda[CT_MAX_BANDS];
  double spectrum_energy[CT_MAX_BANDS];
  double spectrum_noise, spectrum_self_similar, spectrum_texture, spectrum_tau, spectrum_beta;

  // implementation-plan-8.md §4.4/§5.4: each adaptive picker mode's own
  // per-rung diagnostic overlay -- kappa with its Gaussian/structured rails
  // for CT_PICK_STRUCTURE (§5.1, Phase 4), p90/p99 concentration with its
  // noise baseline for CT_PICK_PERCENTILE (§5.2, Phase 5) -- drawn on the
  // same rung axis as spectrum_energy above so a pick is legible instead of
  // magic. Populated by Phase 4/5's own _mode_shape case; unused while
  // picker_mode == CT_PICK_FIXED, which draws no extra overlay at all.
  double mode_overlay[CT_MAX_BANDS];
  int mode_overlay_n;

  // a pick that landed while g->pd was stale (DT_SIGNAL_CONTROL_PICKERDATA_READY
  // is dispatched async -- see color_picker_apply -- so the GUI thread can
  // observe g->pd a preview pass behind the pipe it just raced). Retried by
  // _preview_pipe_finished_retry_pick once a fresh pass actually lands,
  // rather than silently dropped.
  gboolean pick_pending;

  // implementation-plan-8.md §4.1/§4.3: the picker-mode dropbox, placed
  // directly above scale_shift's picker row below. Conf-backed, not
  // params-backed -- read directly off this widget wherever the mode is
  // needed (_color_picker_apply_now, the graph overlay), the same way
  // gain_local_contrast/scale_shift are read through their own params-backed
  // widgets, so there is no separate cached copy to go stale.
  GtkWidget *picker_mode;
} dt_iop_contrast_gui_data_t;


const char *name()
{
  return _("advanced contrast");
}

const char *aliases()
{
  return _("local contrast|texture|clarity|detail enhancement");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self, _("enhance local contrast by boosting details while preserving edges"),
     _("creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_EFFECTS;
}

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

// ---------------------------------------------------------------------------
// measuring the detail level from the ladder: the model and the fit
// ---------------------------------------------------------------------------
//
// §2.2: the ladder is built frame-wide (§2.1's _build_ladder) and queried
// per box through its SAT tables by _measure_box, so the box does not bound
// how far the ladder can reach -- what is left here is only the fit: given a
// rung's (wavelength, energy, weight) triples for the picked box, what
// texture size explains them. §2.3's `_fit_spectrum`, below, is that fit.
//
// detail_level is a low-pass size: the filter radius is 2^-detail_level of the
// image's long edge, and everything finer than that window ends up in the
// high-pass term this module boosts. so "the right detail level" is really the
// question "how large is the texture in this part of the picture?", and that
// is the classic scale-selection problem.
//
// Lindeberg's answer is to look for the scale at which a normalized derivative
// operator responds most strongly. for blob-like structure the operator is the
// scale-normalized Laplacian t^gamma * grad^2 L_t (t = sigma^2), and the
// normalization is what makes responses at different scales comparable at all:
// an unnormalized Laplacian simply decays with scale on any real image, so its
// argmax carries no information.
//
// gamma = 1 is the right exponent here, and not by convention. for an image
// whose power spectrum goes as f^-beta the expected energy of t*grad^2 L_t
// works out as t^(2 - (6-beta)/2) = t^((beta-2)/2): dead flat exactly when
// beta = 2, which is the spectrum natural images actually have. so on a patch
// of featureless fractal texture this measure has no opinion, and any peak it
// does report is a real departure from 1/f^2 -- a texture with a size --
// rather than an artifact of the falling spectrum.
//
// two conveniences make this cheap:
//
//  - a difference of Gaussians approximates the normalized Laplacian:
//    L_(k*sigma) - L_sigma ~ (k-1) * t * grad^2 L_t. k is constant along a
//    geometric ladder, so the mean squared DoG *is* the normalized Laplacian
//    energy up to one constant factor, and no explicit t^gamma appears.
//  - t*grad^2 is invariant under rescaling of the coordinates, so running the
//    same fixed sigma ladder on each level of a Gaussian pyramid produces
//    energies that are directly comparable across octaves -- and costs 4/3 of
//    a single full-resolution pass in total.
//
// the whole thing runs on log2 luminance, which is the space the module's own
// detail extraction works in, so the measurement sees exactly the signal that
// will be boosted -- including the shadow-noise suppression from the noise
// bias slider, since it is measured on the same biased luminance buffer.

// CT_SCALES_PER_OCTAVE / CT_SIGMA_BASE / CT_MAX_OCTAVES / CT_MAX_BANDS are
// declared near dt_iop_contrast_gui_data_t, above -- shared with §2.1's ladder.
// below this much detail there is nothing in the area worth enhancing and so
// nothing to size: 0.008 EV rms is half a percent of luminance, and even at
// this module's largest gain it stays under three percent, which is invisible.
// clear sky sits well below it; the grain of dry asphalt, about the faintest
// thing anyone would pick deliberately, sits comfortably above (~40x margin,
// implementation-plan-2.md §7.1, re-verified against Phase 1-5's ladder on 19
// real crops from a mixed image set -- unchanged). Out-of-focus background is
// *not* reliably below it and that is correct, not a bug: a busy bokeh patch
// (blown highlights, dark blobs) can carry more real broadband energy than a
// deliberately-picked faint texture, so forcing it under this floor would
// also have to swallow real texture and destroy the separation the floor
// exists for.
#define CT_FLAT_RMS_EV 0.008
#define CT_FLAT_ENERGY (CT_FLAT_RMS_EV * CT_FLAT_RMS_EV)
// grid resolution of the model fit's texture size, per octave.
// implementation-plan-3.md §2.2: already finer than the 0.15-octave
// tolerance implementation-plan.md §2 accepts on tau (0.125 octave here,
// against beta's coarse grid, which needed the refinement pass added at
// §2.1) -- this one needs no change, and this comment is that check, not
// a placeholder for one.
#define CT_FIT_STEPS_PER_OCTAVE 8.0
// the fit's other grid: the self-similar spectrum's slope. implementation-
// plan-2.md §7.3: research.md §5.3's figure was for *linear* radiance and
// this ladder measures log2 luminance, whose slope need not match -- the old
// [1.6, 3.0] range pegged 8 of 18 real fits (44%) at one of its own bounds,
// on a 19-crop mixed real-image set. Widened and re-centred from where an
// unconstrained grid search (deliberately over-wide, [0.2, 6.0]) actually
// settled real texture content: clean picks (petals, fur, skin, asphalt,
// foliage, architecture) cluster 1.8-3.4 with no pegging under this range;
// only genuinely flat/ambiguous content (clear sky, an out-of-focus patch --
// exactly what CT_FLAT_ENERGY above is supposed to catch first) still rails
// against the top, which is a property of that content having no real
// texture to fit, not of the range being too narrow for it.
// implementation-plan-3.md §8.1: re-checked with §2.1's refinement pass in
// place, on the twelve Phase 9.2 whole-crop picks (picker-regression/
// findings.md's Phase 2 section table). Real-texture crops land 2.10-3.40,
// margin intact on both bounds, shifting <=0.08 from the coarse-grid values
// above -- inside the coarse grid's own 0.2167 step, i.e. the refinement
// resolving quantisation rather than moving the cluster. The two genuinely
// flat/ambiguous crops (an out-of-focus background, a near-uniform sky)
// still peg at exactly 4.0 under refinement too, exactly as predicted above:
// beta is undetermined on content with no real texture, regardless of grid
// resolution. [1.4, 4.0] still brackets real content, not pegs it.
#define CT_FIT_BETA_MIN 1.4
#define CT_FIT_BETA_MAX 4.0
#define CT_FIT_BETA_STEPS 12  // 13 values, CT_FIT_BETA_MIN .. CT_FIT_BETA_MAX
// rungs this far below the peak are noise, and in log space they would
// otherwise dominate the residual
#define CT_ENERGY_FLOOR 1e-6
// how far an idealized hump is expected to sit from a real texture's ladder,
// as a fraction. it is the floor under every rung's error bar: past a couple
// of hundred independent samples a rung stops getting more trustworthy, so
// extra pixels stop buying it extra weight. implementation-plan-2.md §7.4:
// unchanged, re-verified -- across the same 19-crop set, weighting by
// n_indep (§1.2) instead of a pixel count never produced a fit dominated by
// the fine rungs (no pathological texture/self_similar split, no fit that
// ignored the coarse end); this floor is doing its job against the wider
// weight spread §1.2 introduced.
#define CT_MODEL_ERROR 0.15
// what the fit needs to be worth trusting, and so what decides the smallest
// area that can be measured at all. the hump is located by its flanks, so what
// matters is the *range* of sizes the ladder covers rather than how many rungs
// it took to cover them: one full octave, plus enough rungs to outnumber the
// model's three parameters. implementation-plan-4.md §7.5: with today's
// ladder the finest rung is CT_SIGMA_BASE * CT_GAUSSIAN_SIGMA_FACTOR *
// CT_SIGMA_TO_LAMBDA = 7.09 px, so together these put the floor at
// CT_MIN_SPAN * 7.09 = 14.2 px on the short axis -- the number the refusal
// message below actually quotes. 24 px is a different, unstated claim: the
// worst case *after* _measure_box's outward block rounding (three 8 px
// blocks).
#define CT_MIN_SPAN 2.0
#define CT_MIN_BANDS 4

// §2.3: the exact DoG response (research.md §5.4c) of white noise smoothed
// to sigma_t = sqrt(tau), measured at scale s = sigma^2. reduces to the pure
// sensor-noise response (§5.4a) at tau = 0 -- which is what lets the same
// closed form serve as both the model's noise term and its texture term
// below, differing only in what tau each one carries.
static inline double _dog_shape(const double s, const double tau)
{
  const double k2 = 1.5874010519681994;  // 2^(2/3), the DoG step at CT_SCALES_PER_OCTAVE = 3
  return 1.0 / (2.0 * (k2 * s + tau))
       - 2.0 / ((k2 + 1.0) * s + 2.0 * tau)
       + 1.0 / (2.0 * (s + tau));
}

// research.md §5.4's model: E(s) = N*_dog_shape(s,0) + C*s^((beta-2)/2) +
// A*_dog_shape(s,tau) -- sensor noise, self-similar scene content, and a
// texture with a size, respectively. a single hump plus a floor could
// locate a texture's size but had no way to tell a real one from a patch of
// self-similar content that simply disagreed with beta = 2, nor from a noise
// floor rising into the fine end of the ladder -- both read as "texture".
// N, C, A are amplitudes and so constrained >= 0; tau (texture size^2) and
// beta (spectral slope) are fit by grid search: locate the hump by fitting
// its whole shape in log energy, not by reading off a raw peak, whose
// argmax on a falling spectrum carries no information.
typedef struct _ct_fit_t
{
  double noise, self_similar, texture;  // amplitudes N, C, A -- all >= 0
  double tau, beta;                     // texture size^2, spectral slope
  double texture_peak;                  // texture*max(dog_shape) over the *measured* rungs, in energy units
  double residual;                      // weighted log-space residual, winning grid point
} _ct_fit_t;

// implementation-plan-2.md §6.1: why a pick was refused, so the caller can
// say something specific instead of one message covering "too small" and
// "nothing here" alike.
typedef enum _ct_fit_refusal_t
{
  CT_FIT_REFUSED_SPAN,  // fewer than CT_MIN_BANDS rungs, or less than CT_MIN_SPAN octaves
  CT_FIT_REFUSED_FLAT   // nothing above CT_FLAT_ENERGY at any surviving rung
} _ct_fit_refusal_t;

// non-negative least squares for the model's (up to) three linear
// amplitudes, from the already-weighted normal equations AtA x = Aty.
// init_active marks which of the three are free to fit (§2.3's caller fixes
// the noise term off when it has a prior for it); n <= 3, so a closed-form
// Cramer's-rule solve per active set is simpler and cheaper than a generic
// solver at the thousands of tiny solves the grid search below runs -- drop
// the most negative component and resolve until every free component is
// non-negative or none are left.
static void _nnls3(const double AtA[3][3],
                   const double Aty[3],
                   const gboolean *const restrict init_active,
                   double *const restrict x)
{
  // whiten each column by its own RMS magnitude (sqrt of its AtA diagonal)
  // before solving. the three model terms live at wildly different natural
  // scales -- _dog_shape goes as 1/(s+tau), ~1e4 at the finest rung in the
  // frame-relative s the callers pass, while s^((beta-2)/2) stays within a
  // few orders of magnitude of 1 over a ten-octave ladder -- so a ridge or
  // determinant tolerance sized to matter for one column either does
  // nothing for the others or swamps them outright. normalized, every
  // active diagonal is ~1 and a single small, scale-free ridge and
  // determinant floor are meaningful for all three at once; the solved
  // coefficients are scaled back at the end.
  double norm[3];
  for(int a = 0; a < 3; a++) norm[a] = sqrt(fmax(AtA[a][a], 1e-300));

  double M[3][3], y[3];
  for(int a = 0; a < 3; a++)
  {
    y[a] = Aty[a] / norm[a];
    for(int b = 0; b < 3; b++) M[a][b] = AtA[a][b] / (norm[a] * norm[b]);
  }

  const double ridge = 1e-9;

  gboolean active[3] = { init_active[0], init_active[1], init_active[2] };

  for(int pass = 0; pass < 4; pass++)
  {
    int idx[3], k = 0;
    for(int j = 0; j < 3; j++) { x[j] = 0.0; if(active[j]) idx[k++] = j; }
    if(k == 0) return;

    double sol[3] = { 0.0, 0.0, 0.0 };
    if(k == 1)
    {
      const int a = idx[0];
      const double m = M[a][a] + ridge;
      if(m > 1e-12) sol[0] = y[a] / m;
    }
    else if(k == 2)
    {
      const int a = idx[0], b = idx[1];
      const double m00 = M[a][a] + ridge, m01 = M[a][b], m11 = M[b][b] + ridge;
      const double det = m00 * m11 - m01 * m01;
      if(det > 1e-12)
      {
        sol[0] = (y[a] * m11 - y[b] * m01) / det;
        sol[1] = (m00 * y[b] - m01 * y[a]) / det;
      }
    }
    else  // k == 3
    {
      const double m00 = M[0][0] + ridge, m01 = M[0][1], m02 = M[0][2];
      const double m11 = M[1][1] + ridge, m12 = M[1][2], m22 = M[2][2] + ridge;
      const double c00 = m11 * m22 - m12 * m12;
      const double c01 = m01 * m22 - m12 * m02;
      const double c02 = m01 * m12 - m11 * m02;
      const double det = m00 * c00 - m01 * c01 + m02 * c02;
      if(det > 1e-15)
      {
        const double y0 = y[0], y1 = y[1], y2 = y[2];
        sol[0] = (y0 * c00 - m01 * (y1 * m22 - m12 * y2) + m02 * (y1 * m12 - m11 * y2)) / det;
        sol[1] = (m00 * (y1 * m22 - m12 * y2) - y0 * c01 + m02 * (m01 * y2 - y1 * m02)) / det;
        sol[2] = (m00 * (m11 * y2 - y1 * m12) - m01 * (m01 * y2 - y1 * m02) + y0 * c02) / det;
      }
    }

    int worst = -1;
    double worst_val = -1e-9;  // small negative tolerance against float noise, in the whitened scale
    for(int c = 0; c < k; c++)
      if(sol[c] < worst_val) { worst_val = sol[c]; worst = idx[c]; }

    if(worst < 0)
    {
      for(int c = 0; c < k; c++) x[idx[c]] = fmax(sol[c], 0.0) / norm[idx[c]];
      return;
    }
    active[worst] = FALSE;
  }
}

// evaluate the model at one candidate beta over the full tau grid, updating
// (*best_residual, *best) if this beta's best tau does better than anything
// seen so far. Factored out of _fit_spectrum (implementation-plan-3.md §2)
// so the coarse grid and the refinement pass below can share the one inner
// solve instead of drifting apart.
static void _fit_spectrum_at_beta(const double beta,
                                   const double *const restrict s,
                                   const double *const restrict energy,
                                   const double *const restrict weight,
                                   const int n,
                                   const double peak_e,
                                   const double lo,
                                   const int tau_steps,
                                   const gboolean fix_noise,
                                   const double noise_prior,
                                   const gboolean *const restrict init_active,
                                   double *const restrict best_residual,
                                   _ct_fit_t *const restrict best)
{
  double col_c[CT_MAX_BANDS];  // self-similar column depends only on beta
  for(int i = 0; i < n; i++) col_c[i] = pow(s[i], (beta - 2.0) * 0.5);

  for(int q = 0; q <= tau_steps; q++)
  {
    const double sigma_t = lo * exp2((double)q / CT_FIT_STEPS_PER_OCTAVE);
    const double tau = sigma_t * sigma_t;

    double col_n[CT_MAX_BANDS], col_a[CT_MAX_BANDS];
    for(int i = 0; i < n; i++)
    {
      col_n[i] = _dog_shape(s[i], 0.0);
      col_a[i] = _dog_shape(s[i], tau);
    }

    // three IRLS passes to approximate a log-space fit (research.md §5.5)
    // while keeping every inner solve linear: reweight by 1/E_model^2
    // after each solve, starting from the sampling weights alone.
    double w[CT_MAX_BANDS];
    for(int i = 0; i < n; i++) w[i] = weight[i];

    double x[3] = { 0.0, 0.0, 0.0 };
    for(int irls = 0; irls < 3; irls++)
    {
      double AtA[3][3] = { { 0.0 } };
      double Aty[3] = { 0.0, 0.0, 0.0 };
      for(int i = 0; i < n; i++)
      {
        const double y = fix_noise ? energy[i] - noise_prior * col_n[i] : energy[i];
        const double c[3] = { col_n[i], col_c[i], col_a[i] };
        for(int a = 0; a < 3; a++)
        {
          Aty[a] += w[i] * c[a] * y;
          for(int b = a; b < 3; b++) AtA[a][b] += w[i] * c[a] * c[b];
        }
      }
      AtA[1][0] = AtA[0][1]; AtA[2][0] = AtA[0][2]; AtA[2][1] = AtA[1][2];

      _nnls3(AtA, Aty, init_active, x);

      for(int i = 0; i < n; i++)
      {
        const double e_model =
          (fix_noise ? noise_prior : x[0]) * col_n[i] + x[1] * col_c[i] + x[2] * col_a[i];
        w[i] = weight[i] / fmax(e_model * e_model, CT_ENERGY_FLOOR * CT_ENERGY_FLOOR);
      }
    }

    double residual = 0.0;
    for(int i = 0; i < n; i++)
    {
      const double e_model =
        (fix_noise ? noise_prior : x[0]) * col_n[i] + x[1] * col_c[i] + x[2] * col_a[i];
      const double d = log(fmax(energy[i], peak_e * CT_ENERGY_FLOOR))
                      - log(fmax(e_model, peak_e * CT_ENERGY_FLOOR));
      residual += weight[i] * d * d;
    }

    if(residual < *best_residual)
    {
      // A alone is not comparable to N or C: _dog_shape's column is orders
      // of magnitude away from the self-similar one (see _nnls3), so A's
      // size says nothing about how much energy it explains -- and, at
      // large tau, _dog_shape's near-zero, nearly featureless
      // values over every *measured* rung make the (tau, A) pair almost
      // unidentifiable from self-similar-only data: residual stays flat
      // while A drifts arbitrarily high chasing float-noise-scale
      // "improvement". texture_peak reports what A actually delivers over
      // the rungs this box could measure, in the same energy units
      // peak_e is in, which is what the caller below can honestly compare
      // against.
      double col_a_peak = 0.0;
      for(int i = 0; i < n; i++) col_a_peak = fmax(col_a_peak, col_a[i]);

      *best_residual = residual;
      best->noise = fix_noise ? noise_prior : x[0];
      best->self_similar = x[1];
      best->texture = x[2];
      best->tau = tau;
      best->beta = beta;
      best->texture_peak = x[2] * col_a_peak;
      best->residual = residual;
    }
  }
}

// fit the model to one box's per-rung (sigma, energy, weight) triples.
// noise_prior >= 0 fixes N to that value instead of fitting it (research.md
// §5.5: "prefer fixing N from the block-minimum noise estimate... stabilizes
// everything else"); _measure_box supplies §2.4's _ladder_estimate_noise
// estimate, which is < 0 when no rung had a usable floor, and N then fits
// freely alongside C and A.
//
// returns FALSE, leaving *fit untouched, when there are too few rungs or
// too narrow a span to trust a fit, or nothing above the noise floor
// anywhere in the box -- *reason says which (implementation-plan-2.md
// §6.1), so the caller can say something specific.
static gboolean _fit_spectrum(const double *const restrict sigma,
                              const double *const restrict energy,
                              const double *const restrict weight,
                              const int n,
                              const double noise_prior,
                              _ct_fit_t *const restrict fit,
                              _ct_fit_refusal_t *const restrict reason)
{
  // the smallest area that can be measured at all covers exactly one octave,
  // so this comparison is met on the nose there and is given a rounding's
  // worth of slack rather than being left to turn on an ulp. sigma and
  // lambda differ by the fixed CT_SIGMA_TO_LAMBDA factor, so the ratio this
  // compares is the same either way.
  if(n < CT_MIN_BANDS || sigma[n - 1] < CT_MIN_SPAN * sigma[0] * (1.0 - 1e-9))
  {
    *reason = CT_FIT_REFUSED_SPAN;
    return FALSE;
  }

  double peak_e = 0.0;
  for(int i = 0; i < n; i++) peak_e = fmax(peak_e, energy[i]);
  // nothing there at any scale: a blank sky, a blown highlight, a black frame
  if(peak_e <= CT_FLAT_ENERGY)
  {
    *reason = CT_FIT_REFUSED_FLAT;
    return FALSE;
  }

  double s[CT_MAX_BANDS];  // s = sigma^2, the model's own scale variable
  for(int i = 0; i < n; i++) s[i] = sigma[i] * sigma[i];

  const gboolean fix_noise = noise_prior >= 0.0;
  const gboolean init_active[3] = { !fix_noise, TRUE, TRUE };

  // candidate texture sizes: from half the finest rung to the coarsest one.
  // the top of it puts the hump's peak just past the end of the ladder, as
  // far as the rising flank alone can honestly be pushed.
  const double lo = sigma[0] * 0.5;
  const double hi = sigma[n - 1];
  const int tau_steps = MAX((int)(CT_FIT_STEPS_PER_OCTAVE * log2(hi / lo)), 1);

  double best_residual = DBL_MAX;
  _ct_fit_t best = { 0 };

  const double beta_step = (CT_FIT_BETA_MAX - CT_FIT_BETA_MIN) / (double)CT_FIT_BETA_STEPS;

  for(int bi = 0; bi <= CT_FIT_BETA_STEPS; bi++)
  {
    const double beta = CT_FIT_BETA_MIN + (double)bi * beta_step;
    _fit_spectrum_at_beta(beta, s, energy, weight, n, peak_e, lo, tau_steps, fix_noise, noise_prior,
                           init_active, &best_residual, &best);
  }

  // implementation-plan-3.md §2: the coarse grid's own step is 0.2167 --
  // wider than implementation-plan.md §2's 0.2 acceptance tolerance on beta,
  // and wide enough that the fit cannot land on a real slope and buys the
  // shortfall with a texture term that isn't there: on an exact power law at
  // beta = 2.4 the coarse grid returns 2.483 plus an A large enough to push
  // texture_peak past the 1% of peak_e that decides whether a pick reports
  // a sized texture (_mode_shape's found_texture).
  // One refinement pass over the winner's +-1 coarse step, at 8 sub-steps,
  // takes beta to 0.027 resolution: it recovers 2.402 with texture_peak
  // an order of magnitude *below* the threshold, at every box size, and
  // changes nothing where a real bump exists (sigma_t and A unchanged to
  // three digits). implementation-plan-4.md §7.5: the refinement loop below
  // is -8..8, 17 evaluations, so 30 beta evaluations against 13, about 2.3x
  // the search; a flat dense grid would be 4x for the same answer. (The
  // *resolution* claim above, beta_step/8 = 0.027, is unchanged.)
  if(best_residual < DBL_MAX)
  {
    const double coarse_beta = best.beta;
    for(int ri = -8; ri <= 8; ri++)
    {
      const double beta = coarse_beta + beta_step * (double)ri / 8.0;
      if(beta < CT_FIT_BETA_MIN || beta > CT_FIT_BETA_MAX) continue;
      _fit_spectrum_at_beta(beta, s, energy, weight, n, peak_e, lo, tau_steps, fix_noise, noise_prior,
                             init_active, &best_residual, &best);
    }
  }

  // unreachable unless every residual came out NaN (NaN energies), but a
  // refusal must still say why
  if(best_residual == DBL_MAX)
  {
    *reason = CT_FIT_REFUSED_FLAT;
    return FALSE;
  }
  *fit = best;
  return TRUE;
}

// §2.5/research.md §5.6: S(sigma) = A*G(sigma;tau) + C*sigma^(beta-2) (real
// scene detail) and N(sigma) = N*G(sigma;0) (noise), evaluated in exactly the
// (s = sigma^2) basis _fit_spectrum solved in -- reusing _dog_shape as G is
// what keeps this consistent with the fit rather than introducing a second,
// unrelated normalisation. implementation-plan-2.md §3.2: sigma in, not
// lambda -- every caller now converts at its own call site (§3.3's table),
// not here.
static inline void _ct_fit_eval(const _ct_fit_t *const fit, const double sigma,
                                double *const S, double *const N)
{
  const double s = sigma * sigma;
  *S = fit->texture * _dog_shape(s, fit->tau)
     + fit->self_similar * pow(s, (fit->beta - 2.0) * 0.5);
  *N = fit->noise * _dog_shape(s, 0.0);
}

// implementation-plan-4.md §1.2: _dog_shape(s,tau) is the exact 2D spatial
// covariance of white noise (tau=0), or white noise pre-smoothed to
// correlation length sqrt(tau), filtered by a DoG whose two Gaussian radii
// are fixed one ladder rung apart (s, k2*s) -- it is a special case of the
// general two-radius covariance below, which needs no rung: for any pair of
// Gaussian blur radii (sigma_a, sigma_b), Var[blur_a - blur_b] of that same
// field is K*[1/(sigma_a^2+tau) + 1/(sigma_b^2+tau)] - 4K/(sigma_a^2+
// sigma_b^2+2*tau), and matching this at sigma_b^2 = k2*sigma_a^2 against
// _dog_shape's own closed form fixes K = 1/2 -- verified by direct
// arithmetic and against raw numeric quadrature of the filter
// (picker-regression/harness_v2/dig_calibration_scale.py's own approach,
// generalised off the rung pair) to double-precision agreement. sa, sb are
// s = sigma^2, matching _dog_shape's own convention.
static inline double _ct_dog_shape2(const double sa, const double sb, const double tau)
{
  return 0.5 * (1.0 / (sa + tau) + 1.0 / (sb + tau)) - 2.0 / (sa + sb + 2.0 * tau);
}

// ladder rung ratio in s = sigma^2 (2^(2/CT_SCALES_PER_OCTAVE),
// CT_SCALES_PER_OCTAVE = 3), matching _dog_shape's own local k2 -- needed
// again below since _ct_selfsimilar_shape2 has no single fixed rung to work
// from.
#define CT_LADDER_K2 1.5874010519681994

static inline double _ct_xlnx(const double x)
{
  return (x <= 0.0) ? 0.0 : x * log(x);
}

// implementation-plan-4.md §1.2: the self-similar analogue of
// _ct_dog_shape2 above. fit->self_similar multiplies pow(s,(beta-2)/2) in
// _ct_fit_eval, but that coefficient is not the self-similar spectrum's own
// amplitude C0 -- integrating a w^-beta 2D spectrum against a DoG(sigma_a,
// sigma_b) filter gives C0 * (1/2)*Gamma(1-beta/2) * bracket(sa, sb, e),
// bracket(a,b,e) = a^e+b^e-2*((a+b)/2)^e, e=(beta-2)/2, sa = sigma_a^2 -- and
// _fit_spectrum solved for the coefficient of pow(s,(beta-2)/2), which is
// C0*(1/2)*Gamma(1-beta/2)*bracket(1,k2,beta) (bracket(.) factors out
// s^e exactly, leaving an s-independent, beta-dependent shape term). Dividing
// fit->self_similar by that same shape term recovers C0*(1/2)*Gamma(...)
// without ever evaluating Gamma (it cancels), then multiplying by the
// module band's own bracket(sigma_a,sigma_b,beta) gives that band's
// predicted energy directly.
//
// bracket(a,b,e) is a difference from the linear interpolant of a^e, b^e at
// their mean, so it is exactly zero -- not asymptotically, algebraically --
// at e = 0 (beta = 2, a constant is trivially its own linear interpolant)
// and at e = 1 (beta = 4, ditto for a linear function): both real bounds
// inside [CT_FIT_BETA_MIN, CT_FIT_BETA_MAX], and beta = 4 is not a
// theoretical edge -- flat/ambiguous content pegs there exactly
// (implementation-plan-3.md §8.1). Both branches below are the L'Hopital
// limit of bracket_general/bracket_rung at that e, i.e. their derivatives'
// ratio; sigma_a = 0 (the shelf band) is excluded by the caller before this
// is ever reached, since its own continuum limit does not converge here
// regardless of beta (see _ct_predict_shelf_energy). Verified end-to-end
// (predicted band energy against raw numeric quadrature, log-spaced through
// both e = 0 and e = 1) to agree to <1e-9 relative error away from the
// singularities and to remain continuous through them.
static double _ct_selfsimilar_shape2(const double sa, const double sb, const double beta)
{
  const double e = (beta - 2.0) * 0.5;
  const double k2 = CT_LADDER_K2;

  if(fabs(e) < 1e-4)  // beta near 2.0
  {
    const double num = log(sa * sb) - 2.0 * log((sa + sb) * 0.5);
    const double den = log(k2) - 2.0 * log((1.0 + k2) * 0.5);
    return num / den;
  }
  if(fabs(e - 1.0) < 1e-4)  // beta near 4.0
  {
    const double num = _ct_xlnx(sa) + _ct_xlnx(sb) - (sa + sb) * log((sa + sb) * 0.5);
    const double den = k2 * log(k2) - (1.0 + k2) * log((1.0 + k2) * 0.5);
    return num / den;
  }

  const double num = pow(sa, e) + pow(sb, e) - 2.0 * pow((sa + sb) * 0.5, e);
  const double den = 1.0 + pow(k2, e) - 2.0 * pow((1.0 + k2) * 0.5, e);
  return num / den;
}

// the shape term _ct_selfsimilar_shape2's comment divides out, made
// explicit for the shelf below: (1/2)*Gamma(1-beta/2)*bracket(1,k2,e) is a
// ladder rung's own gain on a w^-beta spectrum, in the normalization
// _ct_dog_shape2 gives the noise and texture terms, so fit->self_similar /
// this is the amplitude that multiplies w^-beta directly in the same units
// fit->noise multiplies 1 and fit->texture multiplies exp(-tau*w^2). It is
// ~0.026-0.034 over the whole beta grid, so leaving it out (as the shelf
// integrand did) under-predicted the self-similar part of the shelf ~35x.
// finite through Gamma's poles at beta = 2 and 4 because bracket vanishes
// there: Gamma(z) = Gamma(z+2)/(z*(z+1)) with z = -e turns the poles into
// 1/(e*(e-1)), and the two branches are the same L'Hopital limits
// _ct_selfsimilar_shape2 takes at those e.
static double _ct_selfsimilar_rung_gain(const double beta)
{
  const double e = (beta - 2.0) * 0.5;
  const double k2 = CT_LADDER_K2, m = (1.0 + k2) * 0.5;
  const double g = 0.5 * tgamma(2.0 - e);  // Gamma(z+2), z = 1 - beta/2 = -e
  if(fabs(e) < 1e-4) return g * (2.0 * log(m) - log(k2)) / (1.0 - e);
  if(fabs(e - 1.0) < 1e-4) return g * (k2 * log(k2) - (1.0 + k2) * log(m)) / e;
  return g * (1.0 + pow(k2, e) - 2.0 * pow(m, e)) / (e * (e - 1.0));
}

#define CT_CALIBRATION_QUAD_POINTS 200

// implementation-plan-4.md §1.2: the finest *measured* band (sigma_a == 0,
// i.e. d-space k == 0 in _query_band_energy -- it has absorbed whatever
// detail is finer than its own outer boundary, per modify_roi_in's
// "unresolvable fine tail: drop") is a shelf, not a bump: its own transfer
// never rolls off at the fine end, so unlike every interior band its
// continuum energy does not converge -- _ct_dog_shape2 and
// _ct_selfsimilar_shape2 above both divide by zero or go unphysical at
// sigma_a = 0 (dig_calibration_scale.py's own second table: 8 to 90x the
// interior bandwidth factor, moving with both beta and sigma_0, no closed
// form fits it). Integrated numerically instead, to the frame-relative
// pixel Nyquist w = pi*long_edge, log-spaced from wmax*1e-4 since the
// transfer's own rolloff scale (~1/sigma_b) and the cutoff can be many
// octaves apart. Verified against direct scipy quadrature to <0.1% relative
// error at CT_CALIBRATION_QUAD_POINTS, including at the realistic extreme
// (a 6000px long edge, the finest surviving band at 7px).
static double _ct_predict_shelf_energy(const _ct_fit_t *const fit,
                                       const double sb, const double wmax)
{
  const double wmin = wmax * 1e-4;
  const double du = log(wmax / wmin) / (double)(CT_CALIBRATION_QUAD_POINTS - 1);
  // fit->noise and fit->texture already are their PSDs' amplitudes in the
  // normalization this integrand uses; fit->self_similar is not, see
  // _ct_selfsimilar_rung_gain
  const double c0 = fit->self_similar / _ct_selfsimilar_rung_gain(fit->beta);

  double total = 0.0, prev = 0.0;
  for(int i = 0; i < CT_CALIBRATION_QUAD_POINTS; i++)
  {
    const double w = wmin * exp((double)i * du);
    const double H = 1.0 - exp(-sb * w * w * 0.5);
    const double psd = fit->noise
                      + fit->texture * exp(-fit->tau * w * w)
                      + c0 * pow(w, -fit->beta);
    // H^2 * P(w) * w [2D polar measure] * w [dw = w du, log-spaced grid]
    const double g = H * H * psd * w * w;
    if(i > 0) total += 0.5 * (g + prev) * du;
    prev = g;
  }
  return total;
}

// implementation-plan-4.md §1.2: predicted energy for a module band
// spanning (sigma_a, sigma_b], frame-relative, against the fitted spectrum
// -- replaces _ct_fit_eval's single-point sample, which implicitly assumed
// the band was as narrow as the ladder rungs the fit was solved against
// (2^(1/CT_SCALES_PER_OCTAVE), a third of an octave) rather than the
// module's own full octave.
static double _ct_predict_band_energy(const _ct_fit_t *const fit,
                                      const double sigma_a, const double sigma_b,
                                      const double long_edge)
{
  const double sb = sigma_b * sigma_b;

  if(sigma_a <= 0.0)
    return _ct_predict_shelf_energy(fit, sb, M_PI * fmax(long_edge, 1.0));

  const double sa = sigma_a * sigma_a;
  const double N = fit->noise * _ct_dog_shape2(sa, sb, 0.0);
  const double A = fit->texture * _ct_dog_shape2(sa, sb, fit->tau);
  const double C = fit->self_similar * _ct_selfsimilar_shape2(sa, sb, fit->beta);
  return N + A + C;
}

// §2.5: which target curve _target_curve writes. CT_TARGET_DEFAULT is the
// picker path's one fixed shape (implementation-plan-6.md §5B.1: plan-6
// §3.7b measured that this photographer's own accepted curves are predicted
// worse by a shape derived from the picked area's own excess than by one
// fixed shape, so the picker's job is "which fixed shape", never "derive a
// shape"). CT_TARGET_EQUALIZE is research.md §5.6's "boost what's weak"
// curve, reached only by the "flatten spectrum" preset (§3.4).
typedef enum _ct_target_mode_t
{
  CT_TARGET_EQUALIZE = 1,
  CT_TARGET_DEFAULT  = 2
} _ct_target_mode_t;

// implementation-plan-8.md §4.1: which *shape* the next pick writes -- a GUI
// preference (conf key below), not a param: it decides what the next pick
// does, not what the pixels do, so a mode change must not create a history
// item or land in a style, and needs no version bump. §4.2's own numbering:
// CT_PICK_FIXED stays entry 0 both because Phase 6's A/B needs an in-app
// control to return to, and because "reset the curve to default from a
// pick" is the one thing the pre-plan-8 picker did that a user might still
// want to keep reaching for.
typedef enum _ct_picker_mode_t
{
  CT_PICK_FIXED      = 0,   // writes the default curve, same as every pre-plan-8 pick
  CT_PICK_STRUCTURE  = 1,   // §5.1: box-wide L2/L1 sparseness per rung, _ct_structure_shape
  CT_PICK_PERCENTILE = 2,   // §5.2: p90/p99 block-RMS concentration per rung, _ct_percentile_shape
} _ct_picker_mode_t;

#define CT_PICKER_MODE_CONF "plugins/darkroom/contrastadv/picker_mode"

// ---------------------------------------------------------------------------
// §2.1: the frame-wide DoG ladder + block energy tables
// ---------------------------------------------------------------------------
//
// research.md §5.1: build the measurement ladder over the whole frame instead
// of inside the picked box, and answer a box query against it in O(1) instead
// of rebuilding the ladder per pick. A band's value at a pixel already
// carries context from a sigma-neighbourhood, so a small box can still report
// on a large-wavelength band -- noisily, not truncated -- which is why the
// per-rung weights the fit uses (§2.3's `_fit_spectrum`, above) matter more
// here than they did before.
//
// `dt_gaussian_blur` (common/gaussian.c) is Deriche's recursive
// approximation (see CT_GAUSSIAN_SIGMA_FACTOR above) whose cost is
// independent of sigma, unlike the explicit mirrored kernels
// the picked-region ladder used to need to bound its own reach. That removes
// the old ladder-length ceiling outright; what still bounds this ladder is
// simply running out of pixels to decimate into, not the cost of a wide blur.
//
// §2.2 republishes the ladder this builds through dt_preview_data_t and
// wires color_picker_apply() to query it synchronously -- see that section,
// below, for the box query and the fit it feeds (§2.3's `_fit_spectrum`,
// above).

#define CT_BLOCK 8            // block-statistics granularity, in level-0 (finest rung) pixels
#define CT_LADDER_MIN_DIM 4   // stop decimating once the next level would be smaller than this

// implementation-plan-8.md §5.2/§5.4: a super-block must average at least
// this many (decimated-level) pixels before its RMS is trustworthy enough to
// feed a p90/p99 concentration read -- below it, Gaussian noise alone gives
// a spuriously low p90/p99 (a single sample's |v| has p90/p99 = 1.645/2.576
// = 0.64). 64 is the plan's own number, matching its q_gauss(64) baseline.
#define CT_SUPERBLOCK_MIN_PIX 64.0

static void _ladder_free(_ct_ladder_t *const ladder)
{
  dt_free_align(ladder->sat2);
  dt_free_align(ladder->sat1);
  dt_free_align(ladder->sat_n);
  dt_free_align(ladder->blockrms);
  memset(ladder, 0, sizeof(_ct_ladder_t));
}

// accumulate one rung's band into the ladder's fixed bw x bh block grid.
// block (bx, by) covers level-0 pixels [bx*CT_BLOCK, (bx+1)*CT_BLOCK) x (...),
// which on this rung's own (possibly decimated) level maps to
// [bx*CT_BLOCK/step, (bx+1)*CT_BLOCK/step) x (...) -- several level pixels
// wide before the ladder has decimated past CT_BLOCK, one or less afterwards.
// using one grid for every rung, rather than a grid sized to each rung's own
// resolution, is what lets a box query be 4 lookups regardless of which rung
// it is asking about (research.md §5.2): the caller never needs to know a
// rung's own resolution to query it.
//
// implementation-plan-4.md §8.2: blkn, if not NULL, gets each block's own
// (x1-x0)*(y1-y0) -- the real level-pixel count this block just summed, 0 for
// a block past the frame edge. bx==bw-1/by==bh-1 (the block grid's own last
// column/row) is the only place this differs from the nominal (CT_BLOCK/
// step)^2 every consumer used to assume instead: width/height are essentially
// never an exact multiple of CT_BLOCK*step, so that block holds fewer real
// pixels than a full one, and every box query touching it was averaging that
// smaller sum over a denominator sized for a full block -- reading the whole
// box's mean low by an amount a whole-frame pick measured at 1-2% on real
// crops (dig_block_edge_norm.c). _ladder_rung_noise_floor reads blkn too,
// to leave those partial blocks out of the frame-wide floor altogether.
static void _ladder_accumulate_blocks(const float *const restrict band,
                                      const size_t cw, const size_t ch,
                                      const double step,
                                      const size_t bw, const size_t bh,
                                      double *const restrict blk2,
                                      double *const restrict blk1,
                                      double *const restrict blkn)
{
  DT_OMP_FOR()
  for(size_t by = 0; by < bh; by++)
  {
    const size_t y0 = MIN((size_t)((double)(by * CT_BLOCK) / step), ch);
    size_t y1 = MIN((size_t)ceil((double)((by + 1) * CT_BLOCK) / step), ch);
    if(y1 <= y0) y1 = MIN(y0 + 1, ch);

    for(size_t bx = 0; bx < bw; bx++)
    {
      double s2 = 0.0, s1 = 0.0;
      size_t n = 0;
      if(y0 < ch)
      {
        const size_t x0 = MIN((size_t)((double)(bx * CT_BLOCK) / step), cw);
        size_t x1 = MIN((size_t)ceil((double)((bx + 1) * CT_BLOCK) / step), cw);
        if(x1 <= x0) x1 = MIN(x0 + 1, cw);

        if(x0 < cw)
        {
          n = (x1 - x0) * (y1 - y0);
          for(size_t j = y0; j < y1; j++)
          {
            const float *const row = band + j * cw;
            for(size_t i = x0; i < x1; i++)
            {
              const double v = (double)row[i];
              s2 += v * v;
              s1 += fabs(v);
            }
          }
        }
      }
      blk2[by * bw + bx] = s2;
      blk1[by * bw + bx] = s1;
      if(blkn) blkn[by * bw + bx] = (double)n;
    }
  }
}

// standard summed-area table, one row/column of zero padding on the low side
// so a box query is sat[y1][x1] - sat[y0][x1] - sat[y1][x0] + sat[y0][x0]
// with no special-casing at the edges. kept in double precision -- the whole
// point of accumulating it once per ladder rather than per pick -- because a
// query can sum thousands of blocks and this is exactly the kind of
// running sum that drifts in float.
//
// implementation-plan-4.md §8.1: measured, unchanged. _ladder_fill_cb/
// _band_fill_cb narrow every published node to float regardless, so the
// question is whether that narrowing (not this accumulation) loses anything
// -- dig_sat_precision.c compared a box query straight off this table in
// double against the same query through a float-narrowed copy, on real
// preview-sized images. Ordinary boxes: unmeasurable (0.0000% at every size
// from 2x2 blocks to the whole frame). A frame's own flattest block can be
// close enough to mathematically flat that the four-corner float subtraction
// catastrophically cancels -- one measured case came back a 1.7e9% relative
// error, far past Issue J.1's own 10%+ guess -- but only because the true
// answer was already ~5-6 orders of magnitude below anything a real pick
// reports (and below CT_FLAT_ENERGY), so the wrong float value is still
// harmlessly small in absolute terms, not mistakable for real texture.
static void _ladder_build_sat(const double *const restrict blk,
                              const size_t bw, const size_t bh,
                              double *const restrict sat)
{
  const size_t sw = bw + 1;
  for(size_t x = 0; x <= bw; x++) sat[x] = 0.0;
  for(size_t y = 1; y <= bh; y++)
  {
    sat[y * sw] = 0.0;
    double rowsum = 0.0;
    for(size_t x = 1; x <= bw; x++)
    {
      rowsum += blk[(y - 1) * bw + (x - 1)];
      sat[y * sw + x] = sat[(y - 1) * sw + x] + rowsum;
    }
  }
}

// implementation-plan-8.md §5.2/§5.4: how many CT_BLOCK base blocks per side
// a super-block groups on a rung whose level is `step` level-0 pixels per
// pixel, so that the group holds at least CT_SUPERBLOCK_MIN_PIX *distinct*
// level pixels. A base block spans CT_BLOCK/step level pixels per side --
// several before the ladder has decimated past CT_BLOCK, a fraction of one
// afterwards, where _ladder_accumulate_blocks' x1<=x0 rescue makes
// neighbouring base blocks re-read the *same* level pixel rather than each
// hold one of its own. So the count is not floored at one per block: past
// step = CT_BLOCK the distinct-pixel count per side is grp * CT_BLOCK / step
// for the whole group, and the group has to grow by another factor of two
// per octave beyond what a one-per-block floor would say (step 16 wants
// grp = 16, not 8; step 64 wants 64, not 8). plan-8 Phase 0 caught the
// floored version: it left every rung from octave 4 up reading p90/p99 off
// groups of 16, 4 and then a single distinct pixel, exactly the case §5.2's
// 64-sample argument exists to rule out. Shared with the offline harness
// (picker-regression/harness_v2/dig_phase0.c) so the two cannot drift.
static size_t _ladder_superblock_side(const double step)
{
  const double per_side = (double)CT_BLOCK / step;  // distinct level pixels per base block side
  size_t grp = 1;
  while((double)(grp * grp) * per_side * per_side < CT_SUPERBLOCK_MIN_PIX && grp < 4096) grp *= 2;
  return grp;
}

// implementation-plan-8.md §5.2/§5.4: aggregate this rung's own blk2/blkn
// (already accumulated by _ladder_accumulate_blocks, same call the SATs
// above are built from) into super-blocks of at least CT_SUPERBLOCK_MIN_PIX
// distinct (decimated-level) pixels, _ladder_superblock_side base blocks per
// side, and write each super-block's RMS (sqrt(sum b^2 / sum n)) into every
// base block it covers -- so a later box query can index this component by
// the same (by, bx) coordinates it already uses for sat2/sat1/sat_n, with no
// per-rung resolution bookkeeping. Using each group's real summed blkn
// rather than a nominal figure makes a frame-edge group -- whose blocks hold
// fewer real pixels than an interior one -- self-correcting: it still divides
// by what it actually summed.
static void _ladder_build_blockrms(const double *const restrict blk2,
                                   const double *const restrict blkn,
                                   const size_t bw, const size_t bh,
                                   const double step,
                                   double *const restrict dst)
{
  const size_t sw = bw + 1;
  const size_t grp = _ladder_superblock_side(step);

  for(size_t gy = 0; gy < bh; gy += grp)
  {
    const size_t y1 = MIN(gy + grp, bh);
    for(size_t gx = 0; gx < bw; gx += grp)
    {
      const size_t x1 = MIN(gx + grp, bw);

      double sum2 = 0.0, sumn = 0.0;
      for(size_t y = gy; y < y1; y++)
        for(size_t x = gx; x < x1; x++)
        {
          sum2 += blk2[y * bw + x];
          sumn += blkn[y * bw + x];
        }
      const double rms = (sumn > 0.0) ? sqrt(sum2 / sumn) : 0.0;

      for(size_t y = gy; y < y1; y++)
        for(size_t x = gx; x < x1; x++)
          dst[y * sw + x] = rms;  // padding row bh / column bw untouched, stays 0
    }
  }
}

// §2.4/research.md §5.2, §5.5: this rung's noise floor -- the minimum block
// energy among blocks whose L2/L1 ratio (kappa) is close to what pure
// Gaussian noise gives. Restricting to near-Gaussian blocks is what keeps a
// block that happens to hold real texture or a hard edge -- either one pushes
// kappa well away from CT_KAPPA_GAUSSIAN -- from dragging the floor down
// below the sensor's actual noise level. Computed frame-wide, over every
// block on the rung, not just a picked box: "noise is global, texture is
// local" (implementation-plan.md §2.4), so this is a far more stable
// estimate than anything one box could produce on its own.
#define CT_KAPPA_GAUSSIAN 1.2533141373155003  // sqrt(pi/2), kappa of Gaussian noise
#define CT_KAPPA_NOISE_TOL 0.10                // +/- 10% of CT_KAPPA_GAUSSIAN counts as "noise-like"
#define CT_KAPPA_EDGE 2.0                      // sparseness warning threshold, research.md §5.2/§5.9
#define CT_NOISE_DOMINATED_FRAC 0.15           // S/(S+N) below this at the box's peak rung -> warn

// implementation-plan-8.md §5.1/Phase 0.1: the kappa value at which a rung
// counts as fully structured for CT_PICK_STRUCTURE -- 90th percentile of
// box-wide kappa_r pooled over 11 real frames x 22 rungs each
// (plan-8-evidence/00-families-on-12-crops.txt §0.1), not the CT_KAPPA_EDGE
// warning rail: real content's kappa rarely reaches 2.0, so anchoring the
// "fully structured" end there would leave the mode permanently weak.
#define CT_KAPPA_STRUCT 1.7999

// implementation-plan-2.md §7.2: unchanged, re-verified against Phase 2's
// ladder on real content, including two deliberately high-ISO/deep-shadow
// crops (where sensor noise should be most visible if this were going to
// break) -- every rung above octave 2 (index >= 3*CT_SCALES_PER_OCTAVE)
// still comes back -1 (no near-Gaussian block found), and
// _ladder_estimate_noise's min-over-rungs still lands on one of the fine
// rungs, giving a small, sane N rather than swamping the fit (§7.4 confirms
// the fits themselves stayed sane downstream of it).
//
// only full blocks take part: a block clipped by the frame edge holds fewer
// than n_per_block level pixels (blkn, see _ladder_accumulate_blocks), and
// dividing its sums by the nominal count does not cancel out of kappa --
// sqrt(e)/m scales as sqrt(n_per_block/n), so a Gaussian block at 90% fill
// still read as noise-like while its energy came out 10% low, dragging a
// minimum-based floor down with it. Smaller partial blocks fell outside the
// kappa tolerance anyway, so skipping them all changes nothing there; using
// their real n instead would let an 8-sample block's noisy energy win the
// minimum, which is worse than leaving it out.
static double _ladder_rung_noise_floor(const double *const restrict blk2,
                                       const double *const restrict blk1,
                                       const double *const restrict blkn,
                                       const size_t nblocks,
                                       const double step)
{
  const double n_per_block = fmax(1.0, (CT_BLOCK / step) * (CT_BLOCK / step));
  double floor_e = -1.0;
  for(size_t i = 0; i < nblocks; i++)
  {
    if(blkn[i] < n_per_block) continue;
    const double m = blk1[i] / n_per_block;
    if(m <= 0.0) continue;
    const double e = blk2[i] / n_per_block;
    const double kappa = sqrt(e) / m;
    if(fabs(kappa - CT_KAPPA_GAUSSIAN) <= CT_KAPPA_NOISE_TOL * CT_KAPPA_GAUSSIAN
       && (floor_e < 0.0 || e < floor_e))
      floor_e = e;
  }
  return floor_e;  // < 0: no near-Gaussian block found on this rung
}

// §2.4: fold every rung's own noise floor into a single scalar estimate of
// the fit's N (research.md §5.5's "block-minimum noise estimate"). Each
// rung's floor is a raw energy; dividing out that rung's own noise transfer
// function (_dog_shape(s,0)) recovers what N it implies, and the minimum
// implied N across the ladder is the least contaminated one -- self-similar
// or textured content only ever adds energy on top of the true noise floor,
// never subtracts from it.
static double _ladder_estimate_noise(const double *const restrict sigma,
                                     const double *const restrict noise_floor,
                                     const int nrungs)
{
  double best = -1.0;
  for(int r = 0; r < nrungs; r++)
  {
    if(noise_floor[r] < 0.0) continue;
    const double g0 = _dog_shape(sigma[r] * sigma[r], 0.0);
    if(g0 <= 0.0) continue;
    const double n_est = noise_floor[r] / g0;
    if(best < 0.0 || n_est < best) best = n_est;
  }
  return best;  // < 0: no usable rung, caller's fit falls back to fitting N freely
}

// build the ladder frame-wide: mean-centre log2 luminance once, then climb
// octaves, each one's four rungs blurred straight from that octave's own
// base (never chained rung-to-rung, which is what keeps every rung an
// honest measurement of the base rather than of an already-smoothed
// approximation of it), decimating by two between octaves once the base is
// safely past its own Nyquist frequency. There is no small-area floor to
// fall back from here, unlike a picked-region ladder: a preview frame is
// always large enough to decimate, so this always does (research.md §5.1).
static gboolean _build_ladder(const float *const restrict lum,
                              const size_t width, const size_t height,
                              _ct_ladder_t *const ladder)
{
  memset(ladder, 0, sizeof(_ct_ladder_t));
  if(width < CT_BLOCK || height < CT_BLOCK) return FALSE;

  ladder->bw = (width + CT_BLOCK - 1) / CT_BLOCK;
  ladder->bh = (height + CT_BLOCK - 1) / CT_BLOCK;
  const size_t sat_stride = (ladder->bw + 1) * (ladder->bh + 1);
  const size_t npixels = width * height;

  double *const restrict sat2 = dt_alloc_align_double(sat_stride * CT_MAX_BANDS);
  double *const restrict sat1 = dt_alloc_align_double(sat_stride * CT_MAX_BANDS);
  double *const restrict sat_n = dt_alloc_align_double(sat_stride * CT_MAX_BANDS);
  // §5.4: same nrungs * sat_stride sizing as sat2/sat1/sat_n, zeroed below so
  // the SAT-style padding row/column (never written below) reads back as 0.
  double *const restrict blockrms = dt_alloc_align_double(sat_stride * CT_MAX_BANDS);
  double *const restrict blk2 = dt_alloc_align_double(ladder->bw * ladder->bh);
  double *const restrict blk1 = dt_alloc_align_double(ladder->bw * ladder->bh);
  double *const restrict blkn = dt_alloc_align_double(ladder->bw * ladder->bh);
  float *restrict level = dt_alloc_align_float(npixels);   // this octave's base
  float *restrict next = dt_alloc_align_float(npixels);    // next octave's decimated base
  float *restrict band = dt_alloc_align_float(npixels);
  float *restrict rung[CT_SCALES_PER_OCTAVE + 1] = { 0 };

  gboolean ok = sat2 && sat1 && sat_n && blockrms && blk2 && blk1 && blkn && level && next && band;
  for(int s = 0; ok && s <= CT_SCALES_PER_OCTAVE; s++)
  {
    rung[s] = dt_alloc_align_float(npixels);
    ok = ok && rung[s];
  }

  if(!ok)
  {
    dt_free_align(sat2); dt_free_align(sat1); dt_free_align(sat_n); dt_free_align(blockrms);
    dt_free_align(blk2); dt_free_align(blk1); dt_free_align(blkn);
    dt_free_align(level); dt_free_align(next); dt_free_align(band);
    for(int s = 0; s <= CT_SCALES_PER_OCTAVE; s++) dt_free_align(rung[s]);
    memset(ladder, 0, sizeof(_ct_ladder_t));
    return FALSE;
  }

  // §5.4: the padding row/column blockrms leaves unwritten below must read
  // back as 0, same as a real SAT's own zero border -- memset once here
  // rather than special-casing it in the per-rung writer.
  memset(blockrms, 0, sat_stride * CT_MAX_BANDS * sizeof(double));

  // mean-centre log2 luminance once, over the whole frame -- everything
  // downstream is a difference of blurs and so ignores this offset
  // mathematically, but centring it keeps the numbers well away from
  // wherever the exposure happens to put the absolute level (the same
  // exposure-invariance argument as `_fit_spectrum`'s log-energy fit,
  // above -- both want the ladder's numbers independent of exposure).
  double mean_ev = 0.0;
  DT_OMP_FOR(reduction(+ : mean_ev))
  for(size_t k = 0; k < npixels; k++)
  {
    level[k] = log2f(fmaxf(lum[k], NORM_MIN));
    mean_ev += level[k];
  }
  mean_ev /= (double)npixels;
  const float offset = (float)mean_ev;
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++) level[k] -= offset;

  size_t cw = width, ch = height;
  double step = 1.0;
  int nrungs = 0;
  const float minv = -1.0e6f, maxv = 1.0e6f;

  // §2: bring octave 0's base up to absolute CT_SIGMA_BASE once, so every
  // octave starts from the same absolute sigma in its own pixel units.
  // Without this, rung[0] of octave 0 is the raw (unblurred) base while
  // every later octave's base already carries sigma = CT_SIGMA_BASE from the
  // previous octave's own rung[CT_SCALES_PER_OCTAVE] (subsampled by two, so
  // still CT_SIGMA_BASE in the new grid's own units) -- octave 0 alone would
  // then measure every band a sqrt(1 + (CT_SIGMA_BASE/sigma_s)^2) too wide.
  // implementation-plan-2.md §2.
  {
    dt_gaussian_t *const gs =
      dt_gaussian_init((int)width, (int)height, 1, &maxv, &minv, CT_SIGMA_BASE,
                       DT_IOP_GAUSSIAN_ZERO);
    if(!gs) ok = FALSE;
    else
    {
      dt_gaussian_blur(gs, level, next);
      dt_gaussian_free(gs);
      memcpy(level, next, npixels * sizeof(float));
    }
  }

  for(int octave = 0;
      ok && octave < CT_MAX_OCTAVES && nrungs + CT_SCALES_PER_OCTAVE <= CT_MAX_BANDS;
      octave++)
  {
    // rung[0] **is** the base (already at absolute CT_SIGMA_BASE); every
    // later rung is an incremental blur from it, not a fresh blur of the
    // base from sigma 0 -- that incremental step is what keeps every
    // octave's rungs at the same absolute sigma the labels claim.
    memcpy(rung[0], level, cw * ch * sizeof(float));
    for(int s = 1; s <= CT_SCALES_PER_OCTAVE; s++)
    {
      const float target = CT_SIGMA_BASE * exp2f((float)s / CT_SCALES_PER_OCTAVE);
      const float inc = sqrtf(target * target - CT_SIGMA_BASE * CT_SIGMA_BASE);
      dt_gaussian_t *const g =
        dt_gaussian_init((int)cw, (int)ch, 1, &maxv, &minv, inc, DT_IOP_GAUSSIAN_ZERO);
      if(!g) { ok = FALSE; break; }
      dt_gaussian_blur(g, level, rung[s]);
      dt_gaussian_free(g);
    }
    if(!ok) break;

    for(int s = 0; s < CT_SCALES_PER_OCTAVE; s++)
    {
      DT_OMP_FOR()
      for(size_t k = 0; k < cw * ch; k++) band[k] = rung[s][k] - rung[s + 1][k];

      _ladder_accumulate_blocks(band, cw, ch, step, ladder->bw, ladder->bh, blk2, blk1, blkn);
      ladder->noise_floor[nrungs] =
        _ladder_rung_noise_floor(blk2, blk1, blkn, ladder->bw * ladder->bh, step);
      _ladder_build_sat(blk2, ladder->bw, ladder->bh, sat2 + (size_t)nrungs * sat_stride);
      _ladder_build_sat(blk1, ladder->bw, ladder->bh, sat1 + (size_t)nrungs * sat_stride);
      _ladder_build_sat(blkn, ladder->bw, ladder->bh, sat_n + (size_t)nrungs * sat_stride);
      _ladder_build_blockrms(blk2, blkn, ladder->bw, ladder->bh, step,
                             blockrms + (size_t)nrungs * sat_stride);

      // implementation-plan-2.md §3.1: label the rung by its own lower-
      // boundary sigma, not the geometric mean of its two rung sigmas -- the
      // DoG's actual peak sits at CT_SIGMA_TO_LAMBDA * sigma_lower (the exact
      // formula, not "within a percent"), which is 1.41x finer than the old
      // geometric-mean label claimed.
      // implementation-plan-3.md §1.2: label the rung with the sigma the blur
      // actually realised, not the one it was asked for -- the requested
      // sigmas above stay as they are, and the factor cancels out of
      // _dog_shape's k2 = 2^(2/3) rung ratio, so only this label moves.
      const double sigma_s = CT_SIGMA_BASE * CT_GAUSSIAN_SIGMA_FACTOR
                             * exp2((double)s / CT_SCALES_PER_OCTAVE);
      ladder->sigma[nrungs] = sigma_s * step;
      ladder->lambda[nrungs] = ladder->sigma[nrungs] * CT_SIGMA_TO_LAMBDA;
      nrungs++;
    }

    const size_t nw = cw / 2, nh = ch / 2;
    if(nw < CT_LADDER_MIN_DIM || nh < CT_LADDER_MIN_DIM) break;

    // the coarsest rung of this octave (index CT_SCALES_PER_OCTAVE, sigma
    // twice the base) is already blurred well past the new grid's Nyquist
    // frequency, so plain subsampling is exact enough -- the same B'
    // principle phase0-nonrecursive-pyramid.md validated, applied here to a
    // pure measurement, not to anything the halo-risk finding was about.
    DT_OMP_FOR()
    for(size_t j = 0; j < nh; j++)
      for(size_t i = 0; i < nw; i++)
        next[j * nw + i] = rung[CT_SCALES_PER_OCTAVE][(2 * j) * cw + 2 * i];

    float *const swap = level; level = next; next = swap;
    cw = nw; ch = nh;
    step *= 2.0;
  }

  ladder->nrungs = nrungs;

  dt_free_align(blk2); dt_free_align(blk1); dt_free_align(blkn);
  dt_free_align(level); dt_free_align(next); dt_free_align(band);
  for(int s = 0; s <= CT_SCALES_PER_OCTAVE; s++) dt_free_align(rung[s]);

  if(!ok || nrungs == 0)
  {
    dt_free_align(sat2); dt_free_align(sat1); dt_free_align(sat_n); dt_free_align(blockrms);
    memset(ladder, 0, sizeof(_ct_ladder_t));
    return FALSE;
  }

  ladder->sat2 = sat2;
  ladder->sat1 = sat1;
  ladder->sat_n = sat_n;
  ladder->blockrms = blockrms;
  return TRUE;
}

// §2.2: dt_preview_data_fill_t for publishing a just-built ladder. Reshapes
// the ladder's rung-major tables (one (bw+1)x(bh+1) grid per rung) into the
// node-major, per-node-interleaved layout dt_preview_data_t expects
// (`components` floats per "pixel", here per SAT node): 4*nrungs floats per
// node, Sum(b^2)/Sum(|b|)/pixel-count/block RMS for rung 0, then rung 1, and
// so on. A cheap reshape, not a rebuild -- the ladder itself was already
// built outside the GUI lock, which is what this fill runs under
// (dt_preview_data_store's contract: fill() must be cheap).
//
// implementation-plan-4.md §8.2: the third component is sat_n, so a box
// query's denominator comes from the same summed-area table as its
// numerator instead of an assumed-full-block count. implementation-plan-8.md
// §5.4: the fourth is blockrms, which is not a SAT (see _ct_ladder_t's own
// comment) but is stored at the identical (bw+1)*(bh+1)-per-rung stride, so
// this reshape needs no special case for it. Every reader (_measure_box,
// _box_block_percentiles, _spectrum_frame_wide*) indexes node (y, x)'s rung
// r as 4*r + component and checks components == 4*nrungs before trusting
// the buffer.
static void _ladder_fill_cb(void *const user_data, float *const buf, const size_t nelems)
{
  const _ct_ladder_t *const ladder = (const _ct_ladder_t *)user_data;
  const size_t sw = ladder->bw + 1, sh = ladder->bh + 1;
  const size_t comps = (size_t)(4 * ladder->nrungs);
  (void)nelems;  // == sw * sh * comps, by construction of the caller's resize

  for(size_t y = 0; y < sh; y++)
    for(size_t x = 0; x < sw; x++)
    {
      float *const dst = buf + (y * sw + x) * comps;
      for(int r = 0; r < ladder->nrungs; r++)
      {
        dst[4 * r]     = (float)ladder->sat2[(size_t)r * sw * sh + y * sw + x];
        dst[4 * r + 1] = (float)ladder->sat1[(size_t)r * sw * sh + y * sw + x];
        dst[4 * r + 2] = (float)ladder->sat_n[(size_t)r * sw * sh + y * sw + x];
        dst[4 * r + 3] = (float)ladder->blockrms[(size_t)r * sw * sh + y * sw + x];
      }
    }
}

// §3.1: per-band block S1/S2 tables for the module's own delivered bands,
// built over the same CT_BLOCK grid the DoG ladder (§2.1) uses so a box query
// is the same 4-lookups-per-band shape -- but always at step = 1 (module
// bands are full resolution by the time they reach here). scratch_* are
// owned by the caller and reused across every band k in turn; sat1/sat2
// hold CT_BANDS separate (bw+1)x(bh+1) tables,
// band-major, one built per k as _decompose_and_accumulate finishes
// computing that band's b_k.
typedef struct _ct_band_tables_t
{
  size_t bw, bh;
  double *sat2, *sat1;      // CT_BANDS * (bw+1) * (bh+1) doubles each
  // implementation-plan-4.md §8.2: real level-pixel count per block, one
  // (bw+1)*(bh+1) table -- unlike sat2/sat1 this is the same for every band
  // (module bands are always step=1 over the same width/height), so it is
  // built once rather than CT_BANDS times.
  double *sat_n;
  float *scratch_full;      // npixels, this band's own b_k
  double *scratch_blk2, *scratch_blk1;  // bw*bh, this band's own blocks
  double *scratch_blkn;                  // bw*bh, this pass's block pixel counts
} _ct_band_tables_t;

// accumulate one band's already-computed b_k (bt->scratch_full) into its own
// slot of bt's SAT tables. Reuses §2.1's block/SAT helpers verbatim -- they
// were already generic over "one band's array + its own step", and a module
// band's step is simply 1.
//
// implementation-plan-4.md §8.2: also rebuilds bt->sat_n every call. It is
// band-independent (step=1, same width/height for every k) so this repeats
// identical work CT_BANDS times, but the block-count pass touches no pixel
// data -- negligible next to the s2/s1 accumulation already happening in the
// same call -- and it keeps this function free of a first-band special case.
static void _band_tables_accumulate(_ct_band_tables_t *const restrict bt,
                                    const int k, const size_t width, const size_t height)
{
  _ladder_accumulate_blocks(bt->scratch_full, width, height, 1.0, bt->bw, bt->bh,
                            bt->scratch_blk2, bt->scratch_blk1, bt->scratch_blkn);
  const size_t sat_stride = (bt->bw + 1) * (bt->bh + 1);
  _ladder_build_sat(bt->scratch_blk2, bt->bw, bt->bh, bt->sat2 + (size_t)k * sat_stride);
  _ladder_build_sat(bt->scratch_blk1, bt->bw, bt->bh, bt->sat1 + (size_t)k * sat_stride);
  _ladder_build_sat(bt->scratch_blkn, bt->bw, bt->bh, bt->sat_n);
}

// §3.1: dt_preview_data_fill_t for _ct_band_tables_t, the same reshape
// _ladder_fill_cb does for the ladder but with CT_BANDS fixed instead of a
// variable nrungs.
//
// implementation-plan-4.md §8.2: one extra trailing component per node,
// bt->sat_n -- shared across every band, so it rides along once per node
// rather than interleaved per band the way sat2/sat1 are.
static void _band_fill_cb(void *const user_data, float *const buf, const size_t nelems)
{
  const _ct_band_tables_t *const bt = (const _ct_band_tables_t *)user_data;
  const size_t sw = bt->bw + 1, sh = bt->bh + 1;
  const size_t comps = (size_t)(2 * CT_BANDS + 1);
  (void)nelems;  // == sw * sh * comps, by construction of the caller's resize

  for(size_t y = 0; y < sh; y++)
    for(size_t x = 0; x < sw; x++)
    {
      float *const dst = buf + (y * sw + x) * comps;
      for(int k = 0; k < CT_BANDS; k++)
      {
        dst[2 * k]     = (float)bt->sat2[(size_t)k * sw * sh + y * sw + x];
        dst[2 * k + 1] = (float)bt->sat1[(size_t)k * sw * sh + y * sw + x];
      }
      dst[2 * CT_BANDS] = (float)bt->sat_n[y * sw + x];
    }
}

// Compute pixel-wise luminance (no boost) and add the noise bias, exactly as
// the detail ladder below expects to see it.
__DT_CLONE_TARGETS__
static inline void compute_luminance(const float *const restrict in,
                                     float *const restrict luminance,
                                     const dt_iop_roi_t *const roi_in,
                                     const dt_iop_contrast_data_t *const d)
{
  const size_t width = (size_t)roi_in->width;
  const size_t height = (size_t)roi_in->height;
  const size_t npixels = width * height;

  luminance_mask(in, luminance, width, height, DT_TONEEQ_NORM_2, 1.0f, 0.0f, 1.0f);
  const float noise_bias = d->noise_bias;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    luminance[k] += noise_bias;
  }
}

// the ladder: every band a direct, full-resolution eigf call against the
// untouched luminance, accumulated as it goes so no band is ever stored
// (research.md §2.4 option A; see phase0-hybrid-pyramid.md for why this is
// the decomposition this module ships).
//
// correction ends up holding sum_k band_master_k * (gain_k - 1) * b_k, in
// EV, with each band's own knee-bent master already folded in (see the loop
// body) but still missing the Wiener gate, a cheap scalar-per-pixel
// operation applied once by the caller rather than here.
//
// b_k is research.md §2.1's incremental band: log2(blur_{k-1}) - log2(blur_k),
// diffed against the *previous* band's own blur (blur_{-1} = the untouched
// luminance), not always against the original -- that is what makes
// sum_k b_k telescope to a single log2(L) - log2(blur_{nbands-1}) highpass
// when every gain is equal, which is what makes DT_CT_MASK_DETAIL a plain
// single-band highpass. A cumulative log2(L) - log2(blur_k) here
// (diffing every band against the original image) does not telescope and
// silently over-boosts whenever more than one band is open at once.
//
// display_band selects what correction ends up holding (Phase 1.6):
// >= 0 writes that one surviving band's raw b_k, instead of accumulating,
// so the per-band mask view can reuse this same pass rather than a second
// traversal; -2 accumulates the unweighted sum of every band's b_k (the
// DETAIL view, with no gain applied); -1 (or
// anything else negative) is the normal gain-weighted accumulate, which
// doubles as the CORRECTION view before the caller's gate.
__DT_CLONE_TARGETS__
static void _decompose_and_accumulate(const float *const restrict lum,
                                      float *const restrict correction,
                                      float *const restrict coarsest,
                                      const size_t width, const size_t height,
                                      const dt_iop_contrast_data_t *const d,
                                      const int display_band,
                                      _ct_band_tables_t *const restrict bt)  // §3.1, NULL to skip
{
  const size_t npixels = width * height;

  memcpy(coarsest, lum, npixels * sizeof(float));
  memset(correction, 0, npixels * sizeof(float));

  float *const restrict log_lum = dt_alloc_align_float(npixels);
  float *const restrict blur = dt_alloc_align_float(npixels);
  if(!log_lum || !blur)
  {
    // correction is already zero and coarsest already holds lum, so the
    // caller falls through to an identity render; only the message is owed
    dt_control_log(_("advanced contrast failed to allocate memory, check your RAM settings"));
    dt_free_align(log_lum);
    dt_free_align(blur);
    return;
  }

  DT_OMP_FOR()
  for(size_t p = 0; p < npixels; p++)
    log_lum[p] = log2f(fmaxf(lum[p], NORM_MIN));

  for(int k = 0; k < d->nbands; k++)
  {
    memcpy(blur, lum, npixels * sizeof(float));
    fast_eigf_surface_blur(blur, width, height, d->sigma[k], d->feathering[k], d->iterations,
                           DT_GF_BLENDING_LINEAR, 1.0f,
                           0.0f, NORM_MIN, 4.0f);

    // implementation-plan-7.md §4.1(d)/§6 Phase 1.3: the per-band knee is
    // baked in here, per band, before bands are summed -- not as a single
    // scalar applied to the sum afterward. That final-scalar structure is
    // exactly what made every earlier candidate discount every band by
    // whichever one band was tightest; folding d->band_master[k] into each
    // band's own contribution before it is added to correction[] is what
    // makes a band with headroom keep its full requested gain.
    const float gain_minus_one = d->gain[k] - 1.0f;
    const float band_scale = gain_minus_one * d->band_master[k];
    const gboolean is_display = (display_band == k);
    const gboolean detail_mode = (display_band == -2);

    DT_OMP_FOR()
    for(size_t p = 0; p < npixels; p++)
    {
      const float log_blur = log2f(fmaxf(blur[p], NORM_MIN));
      const float b_k = log_lum[p] - log_blur;  // log_lum here holds band (k-1)'s own blur, not the original
      if(is_display) correction[p] = b_k;
      else if(detail_mode) correction[p] += b_k;
      else if(display_band < 0) correction[p] += band_scale * b_k;
      if(bt) bt->scratch_full[p] = b_k;
      log_lum[p] = log_blur;  // becomes band (k+1)'s "previous"
    }
    if(bt) _band_tables_accumulate(bt, k, width, height);

    if(k == d->nbands - 1) memcpy(coarsest, blur, npixels * sizeof(float));
  }

  dt_free_align(log_lum);
  dt_free_align(blur);
}

// the Wiener gate, gauged off the coarsest band -- the closest thing this
// module has to a single smoothed reference -- and applied once to the
// whole accumulated correction rather than per band (research.md's noise
// model is about the local signal level, which the coarsest band already
// estimates about as well as any finer one would).
__DT_CLONE_TARGETS__
static inline float _wiener_gate(const float coarsest_pixel, const float noise_bias)
{
  const float noise_power = noise_bias * noise_bias;
  const float combined_power = coarsest_pixel * coarsest_pixel;
  return fmaxf(combined_power - noise_power, 0.0f) / fmaxf(combined_power, NORM_MIN);
}

// Apply the accumulated, gated correction in linear space.
__DT_CLONE_TARGETS__
static inline void apply_correction(const float *const restrict in,
                                    const float *const restrict correction_ev,
                                    float *const restrict out,
                                    const dt_iop_roi_t *const roi_in)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float multiplier = exp2f(correction_ev[k]);
    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
    out[4 * k + 3] = in[4 * k + 3];
  }
}

/*
 Display a mask -- either the correction (what the module is doing to each
 pixel) or a raw band/DETAIL texture view. Output is a grayscale image
 normalized to [0, 1] where 0.5 = no signal, < 0.5 = negative, > 0.5 =
 positive.

 divisor rescales correction_ev before the sigmoid: 1.0 for CORRECTION,
 where the values are already in the module's own EV units, or a
 frame-wide rms (research.md §4.2) for a raw band/DETAIL view, which
 without it reads as flat grey -- those are un-gained EV differences with
 no fixed scale of their own (implementation-plan.md §1.6).
 */
__DT_CLONE_TARGETS__
static inline void display_correction_mask(const float *const restrict correction_ev,
                                           const float divisor,
                                           float *const restrict out,
                                           const dt_iop_roi_t *const roi_in)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float y = correction_ev[k] / divisor;
    const float intensity = y / sqrtf(y * y + 1.0f) * 0.5f + 0.5f; // Smooth mapping to [0, 1]

    for_each_channel(c)
    {
      out[4 * k + c] = intensity;
    }
    out[4 * k + 3] = 1.0f;
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid,
             void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_contrast_data_t *const d = piece->data;
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  // publish which bands survive at this pipe's scale, for the graph's
  // stripe shading (§1.5). keyed on the FULL pipe, not the preview: the
  // preview runs at a fixed thumbnail scale that has nothing to do with the
  // zoom level the graph is meant to describe.
  if(g && self->dev->gui_attached && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL))
  {
    dt_iop_gui_enter_critical_section(self);
    g->nbands = d->nbands;
    dt_iop_gui_leave_critical_section(self);
  }

  const float *const restrict in = (float *const)ivoid;
  float *const restrict out = (float *const)ovoid;

  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const size_t npixels = width * height;

  float *restrict luminance = dt_alloc_align_float(npixels);
  float *restrict correction = dt_alloc_align_float(npixels);
  float *restrict coarsest = dt_alloc_align_float(npixels);

  if(!luminance || !correction || !coarsest)
  {
    // downstream modules read ovoid regardless, so hand them the input
    // rather than an uninitialized buffer
    dt_iop_copy_image_roi(out, in, 4, roi_in, roi_out);
    dt_control_log(_("advanced contrast failed to allocate memory, check your RAM settings"));
    dt_free_align(luminance);
    dt_free_align(correction);
    dt_free_align(coarsest);
    return;
  }

  compute_luminance(in, luminance, roi_in, d);

  // §2.2/§3.1: keep the frame-wide DoG ladder and the module's own per-band
  // block tables current while the module is expanded, on the untiled
  // preview pipe -- a tile is not a whole frame. color_picker_apply()
  // queries both synchronously on the GUI thread; see there for the box
  // query and the fit/calibration they feed.
  const gboolean update_calibration_data =
    g && self->dev->gui_attached && self->expanded
    && dt_pipe_is_preview(piece->pipe) && !piece->pipe->tiling;

  _ct_band_tables_t band_tables = { 0 };
  gboolean have_band_tables = FALSE;
  if(update_calibration_data)
  {
    band_tables.bw = (width + CT_BLOCK - 1) / CT_BLOCK;
    band_tables.bh = (height + CT_BLOCK - 1) / CT_BLOCK;
    const size_t bstride = (band_tables.bw + 1) * (band_tables.bh + 1);
    band_tables.sat2 = dt_alloc_align_double(bstride * CT_BANDS);
    band_tables.sat1 = dt_alloc_align_double(bstride * CT_BANDS);
    band_tables.sat_n = dt_alloc_align_double(bstride);  // §8.2: band-independent, one table
    band_tables.scratch_full = dt_alloc_align_float(npixels);
    band_tables.scratch_blk2 = dt_alloc_align_double(band_tables.bw * band_tables.bh);
    band_tables.scratch_blk1 = dt_alloc_align_double(band_tables.bw * band_tables.bh);
    band_tables.scratch_blkn = dt_alloc_align_double(band_tables.bw * band_tables.bh);
    have_band_tables = band_tables.sat2 && band_tables.sat1 && band_tables.sat_n
                      && band_tables.scratch_full && band_tables.scratch_blk2
                      && band_tables.scratch_blk1 && band_tables.scratch_blkn;
    if(have_band_tables)
    {
      // implementation-plan-4.md §4.2: dt_alloc_align_double does not zero,
      // and _band_tables_accumulate only fills slots 0..d->nbands-1 while
      // _band_fill_cb below publishes all CT_BANDS of them -- d->nbands is
      // 7 on the default preview (§1.1), so a quarter of the published
      // buffer was uninitialized heap. Nothing reads it today
      // (_query_band_energy stops at band_nbands), but §1.1 moved exactly
      // that boundary, so zero it rather than rely on that staying true.
      // sat_n gets the same treatment: _band_tables_accumulate rebuilds it in
      // full on every k, but only if d->nbands > 0 ever calls it at all.
      memset(band_tables.sat2, 0, sizeof(double) * bstride * CT_BANDS);
      memset(band_tables.sat1, 0, sizeof(double) * bstride * CT_BANDS);
      memset(band_tables.sat_n, 0, sizeof(double) * bstride);
    }
  }

  if(update_calibration_data)
  {
    _ct_ladder_t built;
    if(_build_ladder(luminance, width, height, &built))
    {
      // components must be set before dt_preview_data_store()'s resize
      // check runs, and that check only looks at width/height -- so force a
      // resize (by invalidating the stored dimensions) whenever nrungs, and
      // so components, has changed since the last publish.
      dt_iop_gui_enter_critical_section(self);
      // implementation-plan-8.md §5.4: 4 components per rung now (sat2/sat1/
      // sat_n/blockrms), not 3.
      if(g->pd.components != (size_t)(4 * built.nrungs))
      {
        g->pd.components = (size_t)(4 * built.nrungs);
        g->pd.width = 0;
        g->pd.height = 0;
      }
      dt_iop_gui_leave_critical_section(self);

      // implementation-plan-4.md §4.1: dt_preview_data_store commits
      // pd->hash -- which is what makes the buffer read as fresh -- inside
      // its own critical section. Anything written after it describes the
      // new data but becomes visible later, so a pick landing between the
      // two reads new energies against the previous pass's wavelengths,
      // decimation factors and roi long edge. The nrungs-driven components
      // check above catches a changed *rung count*, which is the common
      // case, but not a zoom or crop that keeps the count and moves the
      // labels -- and roi_in's long edge is an L^(beta-2) lever on the
      // answer (§3.1). Describe first, then publish.
      dt_iop_gui_enter_critical_section(self);
      g->ladder_nrungs = built.nrungs;
      memcpy(g->ladder_sigma, built.sigma, sizeof(g->ladder_sigma));
      memcpy(g->ladder_lambda, built.lambda, sizeof(g->ladder_lambda));
      memcpy(g->ladder_noise_floor, built.noise_floor, sizeof(g->ladder_noise_floor));
      g->ladder_roi_in = *roi_in;
      dt_iop_gui_leave_critical_section(self);

      dt_preview_data_store(&g->pd, built.bw + 1, built.bh + 1, piece, _ladder_fill_cb, &built);
    }
    _ladder_free(&built);
  }

  // Phase 1.6: map the GUI's mask selection onto _decompose_and_accumulate's
  // display_band. CORRECTION and the normal (non-display) path are the same
  // -1 accumulate -- CORRECTION is exactly that sum before the gate/master
  // gain below, "what the module is doing". a band index that isn't
  // currently resolvable (>= d->nbands, greyed out in the graph) has no
  // slot in d->sigma/d->gain; fall back to CORRECTION rather than show
  // nothing.
  const gboolean showing_mask =
    g && g->details_display != DT_CT_MASK_OFF && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL);
  int display_band = -1;
  gboolean showing_texture = FALSE;
  if(showing_mask)
  {
    if(g->details_display == DT_CT_MASK_DETAIL) { display_band = -2; showing_texture = TRUE; }
    else if(g->details_display != DT_CT_MASK_CORRECTION && g->details_display < d->nbands)
    {
      display_band = d->nbands - 1 - g->details_display;
      showing_texture = TRUE;
    }
  }

  _decompose_and_accumulate(luminance, correction, coarsest, width, height, d, display_band,
                            have_band_tables ? &band_tables : NULL);

  // §3.1: publish the per-band tables _decompose_and_accumulate just filled,
  // the same way §2.2 publishes the ladder -- band_sigma/band_nbands are
  // this pass's own d->sigma/d->nbands, kept alongside so a later query
  // evaluates the calibration model at the sigmas that actually produced
  // these energies rather than whatever the picker's current params say.
  if(have_band_tables)
  {
    // implementation-plan-4.md §4.1: same race as the ladder's above --
    // band_pd had no guard at all against it (band_nbands/band_sigma moving
    // under a stable component count, unlike the ladder's nrungs-driven
    // resize). Describe first, then publish.
    dt_iop_gui_enter_critical_section(self);
    g->band_nbands = d->nbands;
    memset(g->band_sigma, 0, sizeof(g->band_sigma));
    memcpy(g->band_sigma, d->sigma, sizeof(float) * MIN(d->nbands, CT_BANDS));
    dt_iop_gui_leave_critical_section(self);

    dt_preview_data_store(&g->band_pd, band_tables.bw + 1, band_tables.bh + 1, piece,
                          _band_fill_cb, &band_tables);
  }
  dt_free_align(band_tables.sat2);
  dt_free_align(band_tables.sat1);
  dt_free_align(band_tables.sat_n);
  dt_free_align(band_tables.scratch_full);
  dt_free_align(band_tables.scratch_blk2);
  dt_free_align(band_tables.scratch_blk1);
  dt_free_align(band_tables.scratch_blkn);

  // a band's or DETAIL's raw b_k has no gain applied -- gain could be zero
  // -- and no fixed scale, so gating/scaling it here would be meaningless;
  // it gets its own rms-based normalization below instead. CORRECTION and
  // the real output both want the gate; the per-band master strength is
  // already baked into correction[] by _decompose_and_accumulate above
  // (implementation-plan-7.md §4.1(d)/§6 Phase 1.3 -- it can no longer be a
  // single scalar applied here, since different bands now carry different
  // masters).
  if(!showing_texture)
  {
    DT_OMP_FOR()
    for(size_t k = 0; k < npixels; k++)
    {
      const float gate = _wiener_gate(coarsest[k], d->noise_bias);
      correction[k] *= gate;
    }
  }

  if(showing_mask)
  {
    float divisor = 1.0f;
    if(showing_texture)
    {
      // scale by the displayed buffer's own frame-wide rms (research.md
      // §4.2) -- without it a low-amplitude band reads as flat grey.
      double sum_sq = 0.0;
      DT_OMP_FOR(reduction(+ : sum_sq))
      for(size_t k = 0; k < npixels; k++)
        sum_sq += (double)correction[k] * (double)correction[k];
      const float rms = sqrtf((float)(sum_sq / (double)npixels));
      divisor = fmaxf(3.0f * rms, 1e-3f);
    }

    dt_iop_gui_enter_critical_section(self);
    g->mask_divisor = divisor;
    dt_iop_gui_leave_critical_section(self);

    display_correction_mask(correction, divisor, out, roi_in);
    piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
  }
  else
  {
    apply_correction(in, correction, out, roi_in);
  }

  dt_free_align(luminance);
  dt_free_align(correction);
  dt_free_align(coarsest);
}

// dt_iop_contrast_data_t (sigma[]/gain[]/nbands on top of the params it
// copies) is bigger than dt_iop_contrast_params_t, but without an init_pipe
// of its own the module got develop/imageop.c's default_init_pipe, which
// sizes piece->data at self->params_size -- too small. commit_params below
// (which the module *does* override) then writes past the end of that
// undersized allocation on every call. Silent until whatever happens to sit
// next on the heap gets its chunk header corrupted, then a crash at a much
// later, unrelated free(): found by exporting a real image with the module
// enabled (every band gain != 1, so commit_params actually runs) under gdb --
// SIGABRT in dt_dev_pixelpipe_cleanup_nodes, malloc_printerr "free(): invalid
// next size". Needs its own init_pipe/cleanup_pipe sized to the real struct,
// same pattern as toneequal.c/colorequal.c/atrous.c.
void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_calloc1_align_type(dt_iop_contrast_data_t);
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
}

// implementation-plan-6.md §4.4/§6 Phase 6: Trentacoste, Mantiuk, Heidrich &
// Dufrot's just-objectionable countershading magnitude (EG 2012, "Unsharp
// Masking, Countershading and Halos: Enhancements or Artifacts?"), fit to
// their 1800 judgements as a cubic/linear piecewise in zeta =
// log10(sigma in degrees). lambda is in log10 contrast for a template edge
// of log10-contrast 1; converting a step of log2 amplitude A = log2(10)
// gives the gain ceiling 1 + 2*lambda that _ct_halo_gain_ceiling below
// returns directly.
static double _ct_halo_lambda_obj(const double zeta)
{
  return (zeta <= 0.418)
    ? -0.249 * zeta * zeta * zeta - 0.233 * zeta * zeta + 0.377 * zeta + 0.674
    :  0.048 * zeta + 0.752;
}

// CT_VIEW_DEGREES = 30 -- Trentacoste's tolerance is set by *angular*, not
// pixel, profile width (§4.4), and darktable knows neither print size nor
// viewing distance. Named so the next reader does not mistake this for a
// measurement rather than the fixed assumption it is. §5B.1/§6 Phase 6.3
// predicts this essentially never binds on the picker's own default.
#define CT_VIEW_DEGREES 30.0

// the ceiling on *effective* per-band gain (1 + master*(g_k-1)), from a
// band's own frame-relative sigma (fraction of the long edge) --
// implementation-plan-6.md §5.3/§6 Phase 6.1.
static double _ct_halo_gain_ceiling(const double sigma_frame)
{
  const double zeta = log10(sigma_frame * CT_VIEW_DEGREES);
  return 1.0 + 2.0 * _ct_halo_lambda_obj(zeta);
}

// implementation-plan-7.md §4.1(d)/§6 Phase 1.3: decided candidate -- a
// smooth, per-band knee, chosen over three single-scalar candidates (a
// hard fmin wall, a global soft knee, and a pick-time-only clamp) that all
// scale every band by one number, decided by whichever single band is
// tightest. The ceiling budget varies ~3x across the ladder (§8.2), so any
// single global number either under-uses a loose band's headroom or
// overspends the tight one's. Evaluating the same knee shape per band, off
// that band's own R_k, means a band with headroom is never discounted for a
// different band's constraint -- confirmed on real B&W frames
// (plan-7-evidence/phase1-2-ceiling-renders/) and reasoned through with the
// user directly.
//
// The knee itself is not the plan's original min(x, R*(x/R)^q): that is two
// smooth curves joined by a hard minimum -- continuous in value at R, but
// the slope jumps from q to 1 with no transition, a real corner rather than
// a knee. Replaced by the standard smoothly-broken power law
//   g(x) = x * (1 + (x/R)^s)^((q-1)/s)
// which has the same two asymptotes (slope 1 for x<<R, R*(x/R)^q for x>>R)
// joined smoothly (C-infinity) instead of at a corner. s controls how wide
// the transition is: s -> infinity recovers the old hard corner exactly;
// s=6 keeps the deviation from an unclamped curve under 0.01% at the
// masters every preset and the post-pick default actually use (~1.0-1.5),
// while still giving R itself a real, visible give instead of a slope
// discontinuity.
#define CT_HALO_KNEE_Q 0.5
#define CT_HALO_KNEE_S 6.0

static double _ct_halo_smooth_knee(const double x, const double Rk)
{
  if(x <= 0.0 || Rk == DBL_MAX) return x;
  const double u = pow(x / Rk, CT_HALO_KNEE_S);
  return x * pow(1.0 + u, (CT_HALO_KNEE_Q - 1.0) / CT_HALO_KNEE_S);
}

// implementation-plan-7.md §4.2: the ceiling and the per-band knee it drives
// are both frame-relative -- a function of the band's own param index and
// scale_shift only (§4.4), never of the pipe's roi/zoom. That is what lets
// the graph call these two functions directly from self->params and get
// exactly what process() applies, with no publish/critical-section round
// trip through the pipe: unlike g->nbands (genuinely roi-dependent -- which
// bands survive at this zoom), there is nothing here the GUI cannot already
// derive itself. modify_roi_in below is the other caller, so this is shared
// rather than duplicated.
static double _ct_band_ceiling(const int k, const float scale_shift)
{
  const float D = CT_BAND_D0 + k + 0.5f + scale_shift;
  const double sigma_frame = exp2(-((double)D + 1.0));
  return _ct_halo_gain_ceiling(sigma_frame);
}

static double _ct_band_master(const int k, const float band_gain, const float master,
                              const float scale_shift)
{
  if(band_gain <= 1.0f) return (double)master;  // no boost requested: nothing to bend
  const double ceiling = _ct_band_ceiling(k, scale_shift);
  const double Rk = (ceiling - 1.0) / ((double)band_gain - 1.0);
  return _ct_halo_smooth_knee((double)master, Rk);
}

void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out,
                   dt_iop_roi_t *roi_in)
{
  dt_iop_contrast_data_t *const d = piece->data;

  // node k's nominal wavelength is S * 2^-(D0+k+shift); boundary k sits half
  // an octave finer, at the geometric mean of nodes k and k+1 -- see
  // implementation-plan.md §1.2. Walk from the finest node to the coarsest so
  // that dropping unresolvable bands (sigma < 0.7px at this roi scale) always
  // trims off the front and the last band processed is always the coarsest,
  // which never gets dropped.
  const float S = MAX(piece->iwidth, piece->iheight);
  int nbands = 0;

  // implementation-plan-7.md §4.4/§4.2: each band's own ratio
  // (ceiling - 1)/(g_k - 1), taken by each band's nominal frame-relative
  // sigma -- not only over bands that survive this pass' fine-tail drop
  // below. The ceiling is a property of the *fixed* CT_BANDS ladder, which
  // is frame-relative; which bands survive is a property of the current
  // roi/zoom. Coupling the two made the per-band master (and therefore the
  // rendered image) differ between preview and export at the same master
  // (§2.3) -- a band dropped here for being unresolvable at this roi scale
  // is not thereby exempt from the ceiling it would bind at full
  // resolution. _ct_band_master (above) needs no roi to run, so it is
  // called directly per surviving band below rather than precomputed for
  // every k up front -- unlike a global minimum (which every single-scalar
  // candidate reduced to, discounting every band by whichever one was
  // tightest), each band's own R_k depends on nothing outside that band, so
  // there is nothing to gain by computing it before we know which bands
  // survive.

  for(int k = CT_BANDS - 1; k >= 0; k--)
  {
    const float D = CT_BAND_D0 + k + 0.5f + d->scale_shift;
    const float diameter = exp2f(-D) * S * roi_in->scale;
    const float sigma = 0.5f * (diameter - 1.0f);

    if(nbands == 0 && sigma < 0.7f) continue;  // unresolvable fine tail: drop

    d->sigma[nbands] = sigma;
    d->gain[nbands] = d->band[k];
    d->band_master[nbands] =
      (float)_ct_band_master(k, d->band[k], d->gain_local_contrast, d->scale_shift);

    // §3.3/research.md §2.4: eigf's a = v/(v+eps) saturates toward a = 1 (no
    // blurring at all) as the window grows, so a single global eps leaves
    // coarse bands empty on most ordinary photographs -- phase0-band-
    // energy.md found the coarsest 1-4 of 9 bands going exactly to zero on
    // three of four test images (portrait, sunset, flower), and recommended
    // promoting this from a Phase-3 "only if" to required. Scale eps up
    // with the band's distance from the finest *nominal* node.
    //
    // implementation-plan-4.md §2.1: the ratio this raises to
    // CT_FEATHERING_EXPONENT is meant to be the band's size relative to the
    // finest one -- a property of the ladder, which is fixed. Taking it
    // against the finest *surviving* band made it a property of the zoom
    // level instead: on a 6000 px image the coarsest band's eps changed by
    // 1.51x between fit zoom and 100% (dig_band_alignment.py), so the module
    // rendered differently at different zooms and the preview's bands --
    // the ones §1's calibration measures -- carried a different eps from
    // the ones the export applies. Between two nominal nodes the ratio is
    // exactly 2^(k_ref - k), so the whole thing is one shift of the param
    // index and needs no sigma at all.
    d->feathering[nbands] =
      d->feathering_base * exp2f(CT_FEATHERING_EXPONENT * (float)(CT_BANDS - 1 - k));
    nbands++;
  }
  d->nbands = nbands;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_contrast_params_t *p = (dt_iop_contrast_params_t *)p1;
  dt_iop_contrast_data_t *d = piece->data;

  d->iterations = p->filter_iterations;
  d->gain_local_contrast = p->gain_local_contrast;
  d->noise_bias = p->noise_bias;
  d->scale_shift = p->scale_shift;
  for(int k = 0; k < CT_BANDS; k++)
    d->band[k] = p->band[k];

  // UI feathering is inverted (higher = stricter edge preservation).
  // Adjust the strength based on the number of iterations to maintain a consistent overall effect regardless of iteration count.
  const float default_feathering = 0.2f;  // Base value based on Christian's experiments for a good balance of edge preservation and contrast boost at default settings.
  // §3.3: this is now the *base* value modify_roi_in scales per band
  // (d->feathering[] is what process() actually reads); modify_roi_in runs
  // after commit_params for the same pipe pass and needs d->band[] (also
  // written here) before it can compute a per-band sigma to scale by.
  d->feathering_base = default_feathering * powf(2.0f, -p->edge_protection) / (p->filter_iterations * p->filter_iterations);
}

// implementation-plan-4.md §6.3 (Issue I.5): the graph's tooltip depends only
// on g->nbands, so it belongs where g->nbands changes -- not rebuilt on every
// _area_draw expose, the one place in this file that used to set a tooltip
// from inside a draw handler (atrous.c/colorequal.c/toneequal.c all set
// theirs once in gui_init).
static void _area_set_tooltip(dt_iop_contrast_gui_data_t *g)
{
  gtk_widget_set_tooltip_text
    (GTK_WIDGET(g->area),
     g->nbands < CT_BANDS
     ? _("drag a node to set its band's gain; double-click to reset it;\n"
         "ctrl+click to visualize that band's own detail texture;\n"
         "middle-click for the plain slider list.\n"
         "the graph's floor is 0.2, not 0 -- drag a slider directly to go lower.\n"
         "dashed nodes were extrapolated, not measured, by the last pick.\n"
         "the thin dashed curve is the *effective* gain after the gain\n"
         "slider is applied; a red node/number means that band's effective gain has\n"
         "gone at or below zero, inverting its detail.\n"
         "the dotted curve is a perceptual countershading ceiling: past it,\n"
         "that band's own effective gain bends off instead of climbing\n"
         "further as you raise gain.\n"
         "the shaded bands on the right are too fine to resolve at the\n"
         "current zoom level and have no effect until you zoom in.")
     : _("drag a node to set its band's gain; double-click to reset it;\n"
         "ctrl+click to visualize that band's own detail texture;\n"
         "middle-click for the plain slider list.\n"
         "the graph's floor is 0.2, not 0 -- drag a slider directly to go lower.\n"
         "dashed nodes were extrapolated, not measured, by the last pick.\n"
         "the thin dashed curve is the *effective* gain after the gain\n"
         "slider is applied; a red node/number means that band's effective gain has\n"
         "gone at or below zero, inverting its detail.\n"
         "the dotted curve is a perceptual countershading ceiling: past it,\n"
         "that band's own effective gain bends off instead of climbing\n"
         "further as you raise gain."));
}

// redraw the graph once a pipe has actually run, so its stripe shading
// (g->nbands, published from process() above) and node positions track
// what just got computed rather than the last GUI edit (§1.5; atrous.c's
// _ui_pipe_done does the same for its own frequency histogram).
static void _ui_pipe_done(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(g && !DT_IN_GUI_UPDATE() && self->enabled && self->expanded)
  {
    _area_set_tooltip(g);
    gtk_widget_queue_draw(GTK_WIDGET(g->area));
  }
}

// ---------------------------------------------------------------------------
// implementation-plan-3.md §4.2: the projection grid's own geometry --
// shared by color_picker_apply, init_presets and the graph's x axis so none
// of the three can ever disagree about where the grid's bounds sit.
//
// implementation-plan-6.md §6 Phase 2.1: moved up from just after
// _project_to_bands so that _target_curve's CT_TARGET_DEFAULT branch (below)
// can call _spectrum_lambda_to_raw_x through _ct_sigma_to_node without a
// forward declaration -- these functions were already grouped together in
// the file, so the move carries the whole group rather than cherry-picking
// one function out of it.
// ---------------------------------------------------------------------------

// implementation-plan-2.md §4.1: the nominal per-band boundary sigma,
// frame-relative (sigma / long edge) and finest-first (idx 0), for a given
// scale_shift -- the band ladder is frame-relative by construction (node
// k's nominal wavelength is S * 2^-(D0+k+shift)), so no pixel term is
// needed here the way modify_roi_in's own sigma[] needs one.
//
// implementation-plan-3.md §1.3/implementation-plan-4.md §5.2:
// CT_GAUSSIAN_SIGMA_FACTOR does **not** belong here, and the next person to
// find it will want to sprinkle it in. This sigma is pure geometry -- a
// detail level turned into a frame-relative size, with no dt_gaussian_blur
// behind it to have widened anything. The correction it eventually drives is
// applied by fast_eigf_surface_blur at d->sigma[k], which is a guided
// filter, not Deriche's smoother, so it does not inherit the factor either.
// _build_ladder's two dt_gaussian_init calls are the module's only ones, and
// its rung labels are the only thing the factor applies to.
static void _ct_band_sigma(float *const restrict sigma, const float scale_shift)
{
  int idx = 0;
  for(int k = CT_BANDS - 1; k >= 0; k--)
  {
    const double D = CT_BAND_D0 + k + 0.5 + scale_shift;
    sigma[idx++] = (float)exp2(-(D + 1.0));
  }
}

// implementation-plan-2.md §4.3/implementation-plan-3.md §3.1: the dense
// log-sigma projection grid's own bounds -- two octaves of fine padding
// below band 0 (none of it below the finest band, so sum_k H_k is exactly 1
// out to the grid's fine end) and the coarse end trimmed to the frame's own
// long edge (past which nothing was measured and nothing can be applied,
// rather than the old sigma[CT_BANDS-1]*4.0, which reached 2.62 octaves
// outside the frame).
static void _ct_grid_bounds(const float *const restrict sigma, double *const lo, double *const hi)
{
  *lo = fmax((double)sigma[0], 1e-6) * 0.25;
  *hi = fmin((double)sigma[CT_BANDS - 1] * 4.0, 1.0 / CT_SIGMA_TO_LAMBDA);
}

// implementation-plan-3.md §3.2/§4.3: sum_k H_k(lambda) -- how much of a
// wavelength's energy the nine bands can address between them, in [0,1].
// The same quantity _project_to_bands weights each grid row by (its own
// loop keeps h[k] too, for the A matrix, so it is not simply routed through
// here); this standalone copy is for callers that only need the sum, i.e.
// implementation-plan-3.md §4.3's graph shading.
static double _ct_band_coverage(const double lambda, const float *const restrict sigma,
                                const int nbands)
{
  double sum_h = 0.0;
  for(int k = 0; k < nbands; k++)
  {
    const double sigma_km1 = (k == 0) ? 0.0 : (double)sigma[k - 1];
    const double sigma_k = (double)sigma[k];
    const double hp_km1 = 1.0 - exp(-2.0 * M_PI * M_PI * sigma_km1 * sigma_km1 / (lambda * lambda));
    const double hp_k   = 1.0 - exp(-2.0 * M_PI * M_PI * sigma_k   * sigma_k   / (lambda * lambda));
    sum_h += hp_k - hp_km1;
  }
  return sum_h;
}

// ---------------------------------------------------------------------------
// §3.1: per-band calibration -- research.md §5.8
// ---------------------------------------------------------------------------
//
// The DoG ladder is linear; eigf is not (§5.8's table: as little as 9% of a
// hard edge's local deviation reaches the high-pass at the default edge
// protection). A curve fitted from the linear ladder therefore over-promises
// wherever the picked box has real local contrast, unless the projection
// above is told how much of its own H_k a band actually delivers.

// implementation-plan-2.md §5.1: peak wavelength of an octave-spaced H_k =
// HP_k - HP_{k-1} pair (sigma_k = 2*sigma_km1) -- pi*sqrt(6/ln 4). The finest
// band (sigma_km1 = 0) is a shelf, not a bump (research.md §2.2), so it will
// not land exactly on its node under the formula below; that is correct and
// should be left alone. implementation-plan-4.md §1.2: the general (any
// sigma_km1, sigma_k, not only an octave pair) peak-wavelength formula this
// specialises from used to be needed here to point-sample the fit at a
// band's own peak; §1.2 replaced that point sample with an integral over the
// band's whole transfer, so the general form is gone and this specialised
// constant is what remains, used only to anchor the graph's own x axis.
#define CT_BAND_PEAK_FACTOR 6.5357852

// map a wavelength (in some roi's own pixels, or a frame-relative fraction
// of the long edge if roi_long_edge is 1.0) to a *raw*, unclamped x
// fraction. Node k is drawn at (k+0.5)/CT_BANDS (_graph_curve_from_params);
// band k's own H_k actually peaks at CT_BAND_PEAK_FACTOR * sigma_lower,
// sigma_lower being the next-finer band's own boundary sigma (half of band
// k's own, under §4.1's octave-spaced frame-relative ladder) -- working
// through §4.1's sigma[k] = 2^-(D0+k+1.5) puts that peak at
// 2^-(D0+k+0.5) * (CT_BAND_PEAK_FACTOR/4). Anchoring the axis there,
// instead of at the nominal detail level the old formula used, is what
// makes a rung/preset shape and the node whose H_k actually responds to it
// land at the same x -- the old nominal-level axis drew the spectrum
// 0.7083606 octave toward the coarse end of the band it belonged to
// (implementation-plan-2.md §5.1's own measurement). Verified here as an
// identity against _band_peak_lambda for all nine bands, per §5.1's own
// acceptance criterion; the doc's own inline code snippet has this
// correction term's sign backwards (confirmed by that check -- an additive
// +log2(F/4), not the doc's -log2(F/4)). Ignores scale_shift, exactly as
// the nodes' own fixed screen positions do, so a rung/preset shape and the
// node it nominally corresponds to line up regardless of where scale_shift
// has since moved the *physical* meaning of that node. Shared by §3.2's
// graph overlay and §3.4's analytic preset shapes -- and, unclamped, by
// implementation-plan-3.md §4.2's axis (below), which is what needs to see
// the projection grid's own overhang past the node ladder rather than have
// it piled at a clamped edge.
static double _spectrum_lambda_to_raw_x(const double lambda, const double roi_long_edge)
{
  const double d = -log2(lambda / fmax(roi_long_edge, 1.0)) + log2(CT_BAND_PEAK_FACTOR / 4.0);
  return (d - CT_BAND_D0) / (double)CT_BANDS;
}

// the presets (§3.4) build their target shapes against the node ladder's
// own [0,1] span, clamped -- they have no screen to draw on, so the grid's
// overhang past the ladder is simply not addressable there and clamping it
// to the nearest node is the right behaviour, unchanged from before §4.2.
static float _spectrum_lambda_to_x(const double lambda, const double roi_long_edge)
{
  return CLAMP((float)_spectrum_lambda_to_raw_x(lambda, roi_long_edge), 0.0f, 1.0f);
}

// implementation-plan-6.md §6 Phase 2.1: sigma (frame-relative, i.e. already
// divided by the long edge) to the module's own node axis -- the inverse of
// _spectrum_lambda_to_raw_x's ((k+0.5)/CT_BANDS at node k) convention, so
// _ct_sigma_to_node of the wavelength a band's H_k actually peaks at returns
// that band's own integer node index exactly. Used only to place
// CT_TARGET_DEFAULT's fixed hump (§5B.1) in the same units the graph draws,
// so the shape and the axis cannot disagree.
static double _ct_sigma_to_node(const double sigma)
{
  return _spectrum_lambda_to_raw_x(sigma * CT_SIGMA_TO_LAMBDA, 1.0) * (double)CT_BANDS - 0.5;
}

// ---------------------------------------------------------------------------
// §2.5: from the fit to a target gain curve, and from that curve to nodes
// ---------------------------------------------------------------------------
//
// research.md §5.6: the picker's job stops at *shape*, never strength --
// both modes below write an absolute, master-independent target curve
// (shape[] is the gain curve itself, used by the caller as-is), so dragging
// the master gain afterwards keeps behaving predictably.
//
// implementation-plan-2.md §8.1: CT_TARGET_EQUALIZE is research.md §5.6's
// own bounded absolute curve, clamp((E_ref/S_hat)^alpha) * S/(S+N) -- the
// "flatten spectrum" preset's own expression (§3.4), not a [0,1] boost shape
// with a separate strength knob.
//
// implementation-plan-3.md §5.1: E_ref used to be evaluated at the geometric
// mean of the *projection grid*'s own bounds, which only equalled the band
// ladder's own geometric mean because the grid's padding factors (0.25 and
// 4.0) were reciprocal. §3.1 breaks that coincidence by trimming the grid's
// coarse end to the frame, so E_ref is now derived from the band ladder
// directly (sigma_ref, the caller's sqrt(sigma[0]*sigma[nbands-1])) rather
// than from whatever the grid's bounds happen to be this call -- the curve's
// level must not move just because the grid's bounds do.
//
// implementation-plan-3.md §5: EQUALIZE is a power law in sigma with
// exponent CT_EQUALIZE_ALPHA*(beta-2). Its dynamic range over this grid is
// 2^(alpha*(beta-2)*span), which passes log2(HI/LO) = 3.06 octaves at
// beta ~= 2.68 on today's 11.18-octave grid (span shrank from 12.0 to this
// after §3.1's trim; recompute from the grid's actual bounds if that trim
// ever moves) -- so on any content the fit reads as steeper than about 2.7
// the envelope is reached by arithmetic, not by anything about the picked
// area. A hard CLAMP there does not bound the *shape*, it deletes it: two,
// three, four adjacent bands can land on the same rail. Squash the log gain
// by L*tanh(log_gain/L) instead, with L set independently on each side to
// the rail's own log distance from 1.0 -- |log_gain| << L passes through
// untouched and only the excursions that would otherwise rail get bent. The
// CLAMP below becomes a safety net (tanh's own range keeps it from ever
// firing) rather than a working part of the curve. This also squashes the
// wiener factor, not only the flattening exponent -- deliberate, and the
// variant that measured best against the twelve recorded picks
// (picker-regression/harness_v2/dig_candidate_fixes.c): the alternative
// (squash the exponent alone, let wiener push freely into the rail) leaves a
// noisy pick railing at the floor for an honest reason but with the same
// unreadable graph.
//
// implementation-plan-3.md §8.2: re-checked after §3's trim+weight and this
// section's own soft limit, on the same twelve crops
// (picker-regression/harness_v2/dig_phase5.c): pegged bands 15->8/108, crops
// with any peg 10->6/12, and never the *target curve* itself (dig_phase5.c
// asserts that directly) -- every remaining peg is _project_to_bands' own
// final CLAMP on the least-squares solution below, not this curve railing.
// Three of the eight are one high-beta crop's coarsest bands (recovered
// beta 4.00 -- honest content wanting more cut than any reasonable envelope
// gives without compromising the rest, the residual this phase predicted).
// The other five are single-band pegs on either near-flat content (whose own
// beta pegs per §8.1 -- content the fit cannot resolve to begin with) or an
// otherwise-clean crop taking one real, strong correction at an end band.
// Left at [0.3, 2.5]: widening would give the least-trustworthy crops more
// room precisely where they are least meaningful, and §4.1 already put both
// rails on screen with a settled answer (§4.5) for what a railed node looks
// like, so a genuine peg now reads as legible feedback, not a hidden clamp.
#define CT_EQUALIZE_ALPHA 0.4
#define CT_EQUALIZE_GAIN_LO 0.3
#define CT_EQUALIZE_GAIN_HI 2.5

// implementation-plan-6.md §5B.1/§6 Phase 2.1: the fixed default shape --
// a log-Gaussian hump on the module's own node axis, least-squares fit
// (residual 1.0e-4) against §3.7b's leave-one-out median of twelve accepted
// hand-drawn atrous curves from this project's own photographer, mapped onto
// this axis via harness/atrous_curves.py's node correspondence. Not derived
// from the picked area: §3.7b measured that a shape derived from the frame's
// own excess-over-continuum predicts these same twelve curves *worse* than
// this one fixed shape does (median R^2 0.545 adaptive vs 0.833 fixed), so
// there is nothing here to re-derive per pick.
//
// CT_DEFAULT_NODE = 4.89 -- lambda = 0.97% of the long edge, nominal detail
// level D8.0.
#define CT_DEFAULT_NODE 4.89
// CT_DEFAULT_WIDTH = 1.65 nodes -- the fitted Gaussian's own sigma.
#define CT_DEFAULT_WIDTH 1.65
// implementation-plan-8.md §3.2: CT_DEFAULT_PEAK_EFF is what gets *written*
// into band[] now, at gain_local_contrast's own neutral default of 1.0 --
// there is no post-pick moment any more at which to bump the master
// (init()/_target_curve write the same hump the picker itself writes,
// nothing to distinguish "just picked" from "just enabled"), so the module's
// baseline curve has to carry its own full effective strength directly
// rather than splitting it across a shape amplitude and a raised master.
// 0.30 = the old split's 0.20 * 1.5 (CT_DEFAULT_PEAK * the deleted
// CT_POST_PICK_MASTER below) -- same effective peak band gain of 1.30, same
// three lines of evidence (this photographer's own median accepted peak
// band gain 1.35; §4.4's countershading ceiling at its preferred 0.65
// fraction, 1.40 at the fussiest band; §4.7's acutance saturation, nothing
// worth buying past roughly there). _ct_band_master's smooth knee is
// homogeneous of degree 1 in (master, R_k) together (R_k = (ceiling-1)/
// (band-1) scales by the same 2/3 factor band-1 scales by 1.5x), so
// band_master(k)*(band-1) -- what process() actually applies -- comes out
// identical under the 0.20x1.5 -> 0.30x1.0 split for every master the user
// might already have set before a pick, not just at the neutral default;
// verified algebraically (plan-8 Phase 1.4) rather than only by rendering.
#define CT_DEFAULT_PEAK_EFF 0.30

// implementation-plan-6.md §6 Phase 2.5, decided: the shape's own natural
// skirt at nodes 7-8 (the two finest, which a preview-scale ladder never
// measures, §3.0) is left alone -- not tapered to zero. Decided by rendering
// (per the plan's own instruction, not by argument): a raised-cosine taper
// forcing shape to 0 over [6.5, 8.5] in node space, rendered as a real 9-band
// override against the untapered curve on a 3000px crop of the dog's eye/fur
// (the densest fine detail in either reported frame, and the frame R8's own
// worry is about), differs from the untapered render by less than JPEG
// quantisation noise -- an auto-levelled pixel-difference image shows no
// structure at all, just uniform low-level noise. The reason it barely
// registers: node 8 (the only band the taper meaningfully moves) is already
// down to shape=0.169 (a 1.034 band gain) in the untapered curve, so the
// taper's own maximum effect is removing a ~3% boost from one band. Tapering
// would be a second free parameter (where the cosine starts/ends) bought for
// a difference nobody can see -- simpler code wins.
static void _target_curve(const _ct_fit_t *const fit, const _ct_target_mode_t mode,
                          const double *const restrict sigma_grid, const int m,
                          const double sigma_ref, const float scale_shift,
                          double *const restrict shape)
{
  double s_ref = 0.0, n_ref = 0.0;
  double Lhi = 0.0, Llo = 0.0;
  if(mode == CT_TARGET_EQUALIZE)
  {
    // implementation-plan-4.md §7.4: _ct_fit_eval hands back S and N
    // separately; EQUALIZE's reference level is deliberately S alone, so
    // n_ref itself is unused past this call.
    _ct_fit_eval(fit, sigma_ref, &s_ref, &n_ref);
    Lhi = log(CT_EQUALIZE_GAIN_HI);
    Llo = log(CT_EQUALIZE_GAIN_LO);
  }

  for(int j = 0; j < m; j++)
  {
    if(mode == CT_TARGET_DEFAULT)
    {
      // implementation-plan-6.md §6 Phase 2.4: _ct_sigma_to_node ignores
      // scale_shift by construction (it shares _spectrum_lambda_to_raw_x
      // with the graph's node placement, which is fixed on screen) -- so a
      // given *physical* sigma's node number drifts by scale_shift as the
      // slider moves the ladder under it. Subtracting scale_shift back out
      // anchors the hump to the *ladder's own* node numbering instead: a
      // pick always writes the same nine band[] values regardless of
      // scale_shift, exactly like a hand-drawn curve's band[] does, rather
      // than the hump staying fixed on screen while the physical bands slide
      // under it and the written gains change with the slider.
      const double node = _ct_sigma_to_node(sigma_grid[j]) - (double)scale_shift;
      const double z = (node - CT_DEFAULT_NODE) / CT_DEFAULT_WIDTH;
      shape[j] = 1.0 + CT_DEFAULT_PEAK_EFF * exp(-0.5 * z * z);
      continue;
    }

    double S, N;
    _ct_fit_eval(fit, sigma_grid[j], &S, &N);
    const double wiener = S / fmax(S + N, DBL_MIN);

    double p = CT_EQUALIZE_ALPHA * log(s_ref / fmax(S, DBL_MIN)) + log(fmax(wiener, DBL_MIN));
    p = (p >= 0.0) ? Lhi * tanh(p / Lhi) : Llo * tanh(p / Llo);
    shape[j] = CLAMP(exp(p), CT_EQUALIZE_GAIN_LO, CT_EQUALIZE_GAIN_HI);
  }
}

// §2.5/research.md §5.7: least-squares projection of a target curve onto the
// module's own band gains. Point-sampling g_target at each node's own
// wavelength would be wrong -- the bands overlap too heavily (§2.2) -- so
// this solves for the {g_k} whose H_k basis best reproduces the whole curve
// instead, with a second-difference penalty for smoothness (the fitted model
// is already the regulariser against per-rung noise; this is what keeps nine
// mostly-collinear H_k columns from chasing that noise into an oscillating
// curve).
//
// H_k(lambda) = HP_k(lambda) - HP_{k-1}(lambda), HP_k(lambda) = 1 -
// exp(-2*pi^2*sigma_k^2/lambda^2) is the cumulative fraction of energy at
// wavelength `lambda` band k's own highpass (relative to its boundary sigma)
// would capture; the incremental H_k is what setting gain_k alone adds to
// the module's net response, treating boundary -1 as sigma = 0 (nothing
// captured before band 0).
//
// sigma[] and gains[] both run finest-first (sigma increasing with k), which
// is d->sigma[]'s own convention (§1.2) -- the caller maps back to param
// (coarsest-first) order.
#define CT_PROJECT_GRID 100
// second-difference penalty weight (relative to one grid point's own unit
// weight, scaled by grid size below) -- small enough not to flatten a real
// single-octave hump, large enough that nine mostly-collinear H_k columns
// don't chase per-point noise into an oscillating curve. no closed-form
// value here; picked by eye against a synthetic single-hump target and left
// generous rather than tight, since implementation-plan.md §2.5's own
// acceptance leans on the fit -- not this regulariser -- to keep the curve
// honest.
#define CT_PROJECT_SMOOTHNESS 0.05

// calibration is §3.1's per-band E_module,k/E_predicted,k ratio (NULL, or any
// entry at 1.0, means uncalibrated) -- one scalar multiplier on band k's own
// H_k column, so the solve asks for proportionally more (gain_k - 1) wherever
// eigf's edge-awareness is known to deliver less than this linear model
// would (research.md §5.8). Rows are unaffected: the smoothness penalty acts
// on the {g_k} output directly and has no H_k of its own to calibrate.
//
// implementation-plan-2.md §4.4: gain_lo/gain_hi are the envelope the
// caller's own target curve promised -- research.md §5.6 and _target_curve's
// own comment say the picker sets shape, never strength, so no band should
// come back outside it. The second-difference penalty above can ring past
// the envelope on a two-to-three-band hump (0.974/0.81 measured on real
// picks, both clamp violations, neither a real measurement); this is a hard
// clamp rather than a clamp-and-resolve active set because that ringing was
// small in practice -- revisit if a legitimate curve is measured to flatten
// visibly against it. The picker passes [min(1,master), max(1,master)];
// presets pass their own envelope directly since some (soften) are
// deliberately < 1 and one (flatten spectrum) straddles 1 on both sides,
// neither of which a single "master" scalar can express.
static gboolean _project_to_bands(const double *const restrict lambda_grid,
                                  const double *const restrict g_target,
                                  const int m,
                                  const float *const restrict sigma,
                                  const int nbands,
                                  const float *const restrict calibration,
                                  const float gain_lo, const float gain_hi,
                                  float *const restrict gains)
{
  if(nbands < 2 || m < 2) return FALSE;

  const int nreg = nbands - 2;
  const size_t rows = (size_t)m + (size_t)MAX(nreg, 0);

  float *const restrict A = dt_alloc_align_float(rows * (size_t)nbands);
  float *const restrict y = dt_alloc_align_float(rows);
  if(!A || !y) { dt_free_align(A); dt_free_align(y); return FALSE; }

  memset(A, 0, rows * (size_t)nbands * sizeof(float));

  double sum_rw2 = 0.0;
  for(int j = 0; j < m; j++)
  {
    const double lambda = lambda_grid[j];
    double h[CT_BANDS], sum_h = 0.0;
    for(int k = 0; k < nbands; k++)
    {
      const double sigma_km1 = (k == 0) ? 0.0 : (double)sigma[k - 1];
      const double sigma_k = (double)sigma[k];
      const double hp_km1 = 1.0 - exp(-2.0 * M_PI * M_PI * sigma_km1 * sigma_km1 / (lambda * lambda));
      const double hp_k   = 1.0 - exp(-2.0 * M_PI * M_PI * sigma_k   * sigma_k   / (lambda * lambda));
      h[k] = hp_k - hp_km1;
      sum_h += h[k];
    }
    // implementation-plan-3.md §3.2: sum_k H_k is the fraction of the energy
    // at this wavelength the whole band ladder touches at all -- 1.0 from the
    // fine end through the ladder, then falling away past the coarsest band,
    // where there is simply no band left to respond. §3.1 has already
    // dropped the rows where it is hopeless; these are the ones where it is
    // partial. Unweighted they are answered the only way the solve can: by
    // driving the coarsest band far past what the target asked for.
    // Weighting costs those rows their vote in proportion to how little the
    // ladder can do about them.
    //
    // Scaling both the A row and y by rw makes the *residual* carry weight
    // rw, hence the squared residual rw^2 -- i.e. this is weighted least
    // squares with w_j = sum_h^2, not sum_h.
    const double rw = sum_h;
    sum_rw2 += rw * rw;
    for(int k = 0; k < nbands; k++)
    {
      const float r = calibration ? calibration[k] : 1.0f;
      A[j * nbands + k] = (float)(h[k] * rw) * r;
    }
    y[j] = (float)((g_target[j] - 1.0) * rw);
  }

  // implementation-plan-3.md §3.2/§8.4: CT_PROJECT_SMOOTHNESS was picked by
  // eye against a unit row weight -- the m unweighted rows above each
  // contributed mass 1^2 = 1 to the solve, total mass m. A weighted row now
  // contributes mass rw^2 (its squared residual's own weight, per the
  // comment above), so sum_j rw_j^2 is the same total-mass quantity the
  // constant was tuned against, not m; when weighting is off every rw is 1
  // and sum_rw2 == m, so this is a no-op there. Measured
  // (picker-regression/harness_v2/dig_phase3.c): normalising this way
  // slightly *reduces* the effective smoothness relative to leaving it at m
  // (12 pegged bands / 7 crops on the twelve recovered fits, vs 11 / 7
  // un-normalised) -- the two-to-three-band ringing implementation-plan-2.md
  // §4.4's own comment already documents as the clamp's job, not this
  // penalty's, so a marginally looser penalty here is not a new failure
  // mode, and normalising to what the constant was actually tuned against is
  // the more honest choice.
  const double w = sqrt(CT_PROJECT_SMOOTHNESS * sum_rw2);
  for(int r = 0; r < nreg; r++)
  {
    const int row = m + r;
    A[row * nbands + r]     = (float)w;
    A[row * nbands + r + 1] = (float)(-2.0 * w);
    A[row * nbands + r + 2] = (float)w;
    y[row] = 0.0f;
  }

  const gboolean ok = pseudo_solve(A, y, rows, (size_t)nbands, FALSE);
  if(ok)
    for(int k = 0; k < nbands; k++)
      gains[k] = CLAMP(y[k] + 1.0f, gain_lo, gain_hi);

  dt_free_align(A);
  dt_free_align(y);
  return ok;
}

// implementation-plan-3.md §4.2/implementation-plan-4.md §5.3: the graph's x
// axis is the projection grid -- x0/x1 are that grid's own raw-x bounds
// (coarse/screen-left and fine/screen-right), sourced from the same
// _ct_band_sigma/_ct_grid_bounds color_picker_apply and init_presets now
// call directly, so the axis and the grid it is drawn over cannot disagree.
// Before plan-4 §5.1/§5.2 that was merely true by inspection -- both callers
// open-coded the same formulas beside their own copies of this comment, and
// nothing enforced it. If the grid's bounds ever move, re-derive from here
// rather than copying new numbers in.
typedef struct _ct_axis_t { double x0, x1; } _ct_axis_t;

static _ct_axis_t _graph_axis(const dt_iop_contrast_params_t *const p)
{
  float sigma[CT_BANDS];
  _ct_band_sigma(sigma, p->scale_shift);
  double lo, hi;
  _ct_grid_bounds(sigma, &lo, &hi);
  const _ct_axis_t axis = { _spectrum_lambda_to_raw_x(hi * CT_SIGMA_TO_LAMBDA, 1.0),
                            _spectrum_lambda_to_raw_x(lo * CT_SIGMA_TO_LAMBDA, 1.0) };
  return axis;
}

// a raw x fraction (node space, (k+0.5)/CT_BANDS for node k, or the output
// of _spectrum_lambda_to_raw_x for a wavelength) run through the axis to a
// screen x fraction -- every other mapping below is this plus a conversion
// to raw x.
static float _graph_raw_to_x(const double raw, const _ct_axis_t *const axis)
{
  return CLAMP((float)((raw - axis->x0) / fmax(axis->x1 - axis->x0, 1e-9)), 0.0f, 1.0f);
}

// map a wavelength to a screen x fraction on the given axis -- the drawing
// counterpart of _spectrum_lambda_to_x above, used everywhere the graph
// actually renders (implementation-plan-3.md §4.2).
static float _graph_lambda_to_x(const double lambda, const double roi_long_edge,
                                const _ct_axis_t *const axis)
{
  return _graph_raw_to_x(_spectrum_lambda_to_raw_x(lambda, roi_long_edge), axis);
}

// a node's own fixed raw-x position, (k+0.5)/CT_BANDS, run through the same
// axis -- keeps node placement, the curve and the spectrum overlay all on
// the same map.
static float _graph_node_x(const int k, const _ct_axis_t *const axis)
{
  return _graph_raw_to_x(((double)k + 0.5) / (double)CT_BANDS, axis);
}

// query §3.1's per-band block tables (built alongside the ladder in the same
// guarded preview pass -- see process()'s comment above the ladder build)
// for the box the picker used. e_module/sigma_out are filled finest-first,
// for the first *nbands_out entries only -- sigma_out is band_sigma as of the
// pass that produced e_module, not the picker's possibly-since-changed
// current node placement, so the two stay internally consistent with each
// other even if they drift a little from "right now".
//
// Returns FALSE (leaving every output untouched) if there is nothing fresh
// to offer -- module just opened, or a param change raced the preview pipe.
// Calibration is a refinement on an already-working picker, not a
// precondition for one (research.md §5.10's framing for the analogous
// second-picker question) -- the caller falls back to uncalibrated rather
// than stalling or refusing the pick over a missing table.
static gboolean _query_band_energy(dt_iop_module_t *self, const int *const box,
                                   double *const restrict e_module,
                                   double *const restrict sigma_out,
                                   int *const restrict nbands_out)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;
  if(!dt_preview_data_is_fresh(&g->band_pd)) return FALSE;

  dt_iop_gui_enter_critical_section(self);

  const size_t sat_w = g->band_pd.width, sat_h = g->band_pd.height;
  const gboolean have_data =
    g->band_pd.buf && sat_w > 1 && sat_h > 1
    && g->band_pd.components == (size_t)(2 * CT_BANDS + 1) && g->band_nbands > 0;

  if(have_data)
  {
    const size_t bw = sat_w - 1, bh = sat_h - 1;
    const size_t bx0 = MIN(bw, (size_t)MAX(box[0], 0) / CT_BLOCK);
    size_t bx1 = MIN(bw, (size_t)(MAX(box[2], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(bx1 <= bx0) bx1 = MIN(bw, bx0 + 1);
    const size_t by0 = MIN(bh, (size_t)MAX(box[1], 0) / CT_BLOCK);
    size_t by1 = MIN(bh, (size_t)(MAX(box[3], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(by1 <= by0) by1 = MIN(bh, by0 + 1);

    const int nbands = MIN(g->band_nbands, CT_BANDS);
    const size_t comps = (size_t)(2 * CT_BANDS + 1);
    const float *const restrict buf = g->band_pd.buf;
    // implementation-plan-4.md §8.2: n_eff from the same box query as s2,
    // over the real level-pixel count sat_n carries -- not
    // (bx1-bx0)*(by1-by0)*CT_BLOCK^2, which assumed every block in the box
    // was a full, unclipped CT_BLOCK x CT_BLOCK square. The block grid's own
    // last column/row essentially never is (width/height are essentially
    // never an exact multiple of CT_BLOCK), so that assumption read every
    // box touching the frame edge low by an amount a whole-frame pick
    // measured at 1-2% on real crops (dig_block_edge_norm.c).
    const double n_eff = fmax((double)buf[(by1 * sat_w + bx1) * comps + 2 * CT_BANDS]
                             - (double)buf[(by0 * sat_w + bx1) * comps + 2 * CT_BANDS]
                             - (double)buf[(by1 * sat_w + bx0) * comps + 2 * CT_BANDS]
                             + (double)buf[(by0 * sat_w + bx0) * comps + 2 * CT_BANDS], 1.0);
    for(int k = 0; k < nbands; k++)
    {
      const double s2 = buf[(by1 * sat_w + bx1) * comps + 2 * k]
                       - buf[(by0 * sat_w + bx1) * comps + 2 * k]
                       - buf[(by1 * sat_w + bx0) * comps + 2 * k]
                       + buf[(by0 * sat_w + bx0) * comps + 2 * k];
      e_module[k] = s2 / n_eff;
      sigma_out[k] = (double)g->band_sigma[k];
    }
    *nbands_out = nbands;
  }

  dt_iop_gui_leave_critical_section(self);
  return have_data;
}

// calibration[k] = sqrt(E_module,k / E_predicted,k), clamped against a
// near-empty band's E_predicted blowing the ratio up rather than trusted at
// face value -- this is an empirical correction, not a physical law, and
// both the box and the fit are noisy. calibration defaults every band to 1
// (uncalibrated) first, so a stale or missing table just skips the
// refinement instead of failing the pick.
//
// implementation-plan-4.md §1.3: _project_to_bands multiplies calibration[k]
// onto H_k, an *amplitude* column (its RHS is g_target - 1, a gain) -- so
// what multiplies it has to be an amplitude fraction, not the energy ratio
// above. sqrt() is the conversion; without it a band delivering 62% of its
// amplitude (research.md §5.8's +-0.5 EV texture row) read as 0.38 and the
// solve asked for 2.6x the correction it should have. In amplitude terms a
// physical bound exists -- 1 - a = eps/(v+eps) <= 1 always -- so the upper
// bound sits just above 1 (headroom for the fit's own beta not exactly
// matching a band, not a hard clamp there) rather than at the old energy-
// space value tuned against a ratio whose neutral point was never 1.
#define CT_CALIBRATION_MIN 0.05
#define CT_CALIBRATION_MAX 1.2
#define CT_CALIBRATION_FLOOR 1e-9

static void _compute_band_calibration(dt_iop_module_t *self, const int *const box,
                                      const float long_edge,
                                      const _ct_fit_t *const fit,
                                      float *const restrict calibration)
{
  for(int k = 0; k < CT_BANDS; k++) calibration[k] = 1.0f;

  double e_module[CT_BANDS], sigma_d[CT_BANDS];
  int nbands = 0;
  if(!_query_band_energy(self, box, e_module, sigma_d, &nbands)) return;

  // implementation-plan-4.md §1.1: _query_band_energy answers in d-space --
  // index 0 is the finest band that *survived* modify_roi_in on the pass that
  // built the tables -- while _project_to_bands indexes the full nine-node
  // ladder, index 0 being the finest node whether it survived or not. d index
  // i is param index nbands-1-i, hence projection index CT_BANDS-nbands+i.
  // The two coincide only at nbands == CT_BANDS, which the preview pipe (the
  // only pipe that builds these tables) essentially never reaches: its input
  // is capped at DT_MIPMAP_3/4 (mipmap_cache.c), so nbands is 7 at the default
  // 1440 px long edge and 8 with highres_preview_mip. The calibration was
  // being applied two bands too fine, leaving the coarsest two -- where eigf
  // saturates most and the correction is most needed -- uncalibrated.
  // nbands is MIN(g->band_nbands, CT_BANDS) so offset >= 0 by construction;
  // clamped rather than trusted.
  const int offset = MAX(0, CT_BANDS - nbands);

  // implementation-plan-4.md §1.4: this is what turns the plan's own
  // acceptance criteria into a measurement instead of an argument -- every
  // dig_*.c driver calls _project_to_bands with calibration = NULL, so
  // nothing offline ever exercised this path; a live pick with -d picker is
  // the only way to read r_k, e_module,k and e_predicted,k off together.
  dt_print(DT_DEBUG_PICKER,
           "[contrastadv] calibration nbands=%d offset=%d fit N=%.4g A=%.4g C=%.4g tau=%.4g beta=%.3f",
           nbands, offset, fit->noise, fit->texture, fit->self_similar, fit->tau, fit->beta);

  for(int k = 0; k < nbands; k++)
  {
    // implementation-plan-4.md §1.2: d-space k == 0 is the finest *measured*
    // band regardless of offset -- it has absorbed whatever detail is finer
    // than its own outer boundary (modify_roi_in's "unresolvable fine tail:
    // drop"), so it is a shelf, not a bump, and its continuum energy runs to
    // the pixel Nyquist rather than converging. _ct_predict_band_energy
    // integrates it directly (sigma_km1 = 0 signals the shelf case to it).
    const double sigma_km1 = (k == 0) ? 0.0 : sigma_d[k - 1] / (double)long_edge;
    const double sigma_k = sigma_d[k] / (double)long_edge;
    const double e_predicted
      = fmax(_ct_predict_band_energy(fit, sigma_km1, sigma_k, long_edge), CT_CALIBRATION_FLOOR);
    const double r = e_module[k] / e_predicted;  // energy ratio, E_module,k / E_predicted,k

    calibration[offset + k] = (float)CLAMP(sqrt(fmax(r, 0.0)), CT_CALIBRATION_MIN, CT_CALIBRATION_MAX);

    dt_print(DT_DEBUG_PICKER,
             "[contrastadv]   k=%d (proj %d) e_module=%.4g e_predicted=%.4g r=%.4g calibration=%.4f",
             k, offset + k, e_module[k], e_predicted, r, (double)calibration[offset + k]);
  }
}

// implementation-plan-8.md §5: what a box query reads off the ladder's own
// SAT tables, before any picker mode decides what to do with it -- the
// shared skeleton's `_measure_box` output. Same fields the pre-plan-8
// `_fit_curve_from_box` kept as locals (see its comment, preserved below on
// `_measure_box`), gathered into one struct so `_fit_spectrum` and
// `_mode_shape` can both take a single argument instead of the same eight
// arrays threaded through by hand.
typedef struct _ct_box_stats_t
{
  int nrungs;
  double lambda[CT_MAX_BANDS];
  double sigma[CT_MAX_BANDS];
  double energies[CT_MAX_BANDS];
  double weights[CT_MAX_BANDS];
  double s1_energy[CT_MAX_BANDS];    // §2.4: Sum(|b|)/n_eff over the box, for its own sparseness
  double noise_floor[CT_MAX_BANDS];  // §2.4: frame-wide, not the box's own
  double noise_prior;
  double peak_e;
  double ladder_lambda0;      // §6.1: finest rung's own wavelength
  double window_lambda_max;   // §7: the window's own achievable span
} _ct_box_stats_t;

// §2.2: query the ladder's published SAT tables for the box the picker
// selected, into the per-rung (wavelength, energy, weight) triples every
// picker mode's `_fit_spectrum` call (run separately by the caller, plan-8
// §5's shared skeleton) and `_mode_shape` then share. box is in the pixels
// of g->ladder_roi_in, i.e. of the preview the ladder was built from.
//
// research.md §5.2: any box query is 4 lookups per rung -- this is that
// query, one held critical section covering every rung so the buffer can't
// be resized out from under it mid-query.
//
// returns FALSE only when the ladder itself has nothing published yet (a
// fresh module, or a preview pass still catching up) -- the "too small"/
// "nothing to measure" refusals are `_fit_spectrum`'s own, once the caller
// runs it against *stats. long_edge (§4.2) is the ladder roi's own long
// edge, in the same pixels as the ladder's sigma -- dividing by it is what
// makes the eventual fit->tau frame-relative.
static gboolean _measure_box(dt_iop_module_t *self, const int *const box,
                             const float long_edge, _ct_box_stats_t *const stats)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  memset(stats, 0, sizeof(*stats));

  dt_iop_gui_enter_critical_section(self);

  const size_t sat_w = g->pd.width, sat_h = g->pd.height;
  const size_t comps = g->pd.components;
  const gboolean have_data =
    g->pd.buf && sat_w > 1 && sat_h > 1 && g->ladder_nrungs > 0
    && comps == (size_t)(4 * g->ladder_nrungs);

  if(have_data)
  {
    const size_t bw = sat_w - 1, bh = sat_h - 1;
    const size_t bx0 = MIN(bw, (size_t)MAX(box[0], 0) / CT_BLOCK);
    size_t bx1 = MIN(bw, (size_t)(MAX(box[2], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(bx1 <= bx0) bx1 = MIN(bw, bx0 + 1);
    const size_t by0 = MIN(bh, (size_t)MAX(box[1], 0) / CT_BLOCK);
    size_t by1 = MIN(bh, (size_t)(MAX(box[3], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(by1 <= by0) by1 = MIN(bh, by0 + 1);

    // §1.1: a rung whose wavelength does not fit inside the box at least once
    // is not measuring the box's own texture -- past that size the block sum
    // is dominated by the box's offset from its surroundings (a bias, not
    // noise) and the fit will lock onto that instead. One full period is the
    // loosest defensible cut; see implementation-plan-2.md §1.1.
    const double box_w = (double)(bx1 - bx0) * CT_BLOCK;
    const double box_h = (double)(by1 - by0) * CT_BLOCK;
    const double lambda_max = fmin(box_w, box_h);
    stats->window_lambda_max = lambda_max;
    // §6.1: the ladder's own finest rung, independent of the window above --
    // needed even when the box is too small to keep a single rung, to quote
    // the smallest box that would have worked.
    stats->ladder_lambda0 = g->ladder_lambda[0];

    const float *const restrict buf = g->pd.buf;
    int nrungs = 0;
    for(int r = 0; r < g->ladder_nrungs; r++)
    {
      if(g->ladder_lambda[r] > lambda_max) break;  // rungs run fine -> coarse

      const double s2 = buf[(by1 * sat_w + bx1) * comps + 4 * r]
                       - buf[(by0 * sat_w + bx1) * comps + 4 * r]
                       - buf[(by1 * sat_w + bx0) * comps + 4 * r]
                       + buf[(by0 * sat_w + bx0) * comps + 4 * r];
      const double s1 = buf[(by1 * sat_w + bx1) * comps + 4 * r + 1]
                       - buf[(by0 * sat_w + bx1) * comps + 4 * r + 1]
                       - buf[(by1 * sat_w + bx0) * comps + 4 * r + 1]
                       + buf[(by0 * sat_w + bx0) * comps + 4 * r + 1];

      // implementation-plan-4.md §8.2: n_eff from the same box query as s2/
      // s1, over the real level-pixel count sat_n carries -- not
      // nblocks*(CT_BLOCK/step)^2, which assumed every block in the box was a
      // full, unclipped square. The block grid's own last column/row
      // essentially never is (width/height are essentially never an exact
      // multiple of CT_BLOCK*step), so that assumption read every box
      // touching the frame edge low by an amount a whole-frame pick measured
      // at 1-2% on real crops (dig_block_edge_norm.c).
      const double n_eff = fmax(buf[(by1 * sat_w + bx1) * comps + 4 * r + 2]
                              - buf[(by0 * sat_w + bx1) * comps + 4 * r + 2]
                              - buf[(by1 * sat_w + bx0) * comps + 4 * r + 2]
                              + buf[(by0 * sat_w + bx0) * comps + 4 * r + 2], 1.0);
      const double lam = g->ladder_lambda[r];

      // §1.2: n_eff is a pixel count and is the wrong denominator for the
      // sampling term in the weight -- research.md §5.1's own error table is
      // written in terms of area/sigma^2, i.e. independent samples of the
      // rung's own period, not level pixels. Keep n_eff for the energy mean
      // (s2/n_eff is correct there) and use n_indep only for the weight.
      const double n_indep = fmax(box_w * box_h / (lam * lam), 0.25);

      stats->lambda[nrungs] = lam;
      // §4.2: frame-relative, so it lines up with §4.1's band sigma
      stats->sigma[nrungs] = g->ladder_sigma[r] / (double)long_edge;
      stats->energies[nrungs] = s2 / n_eff;
      stats->s1_energy[nrungs] = s1 / n_eff;
      stats->weights[nrungs] = 1.0 / (CT_MODEL_ERROR * CT_MODEL_ERROR + 2.0 / n_indep);
      stats->noise_floor[nrungs] = g->ladder_noise_floor[r];

      nrungs++;
    }
    stats->nrungs = nrungs;
  }

  dt_iop_gui_leave_critical_section(self);

  if(!have_data) return FALSE;

  for(int r = 0; r < stats->nrungs; r++) stats->peak_e = fmax(stats->peak_e, stats->energies[r]);

  // §2.4: fix N from the frame-wide block-minimum estimate rather than
  // fitting it freely -- "stabilises everything else" (research.md §5.5) and
  // stops a genuinely fine texture from getting explained away as noise.
  stats->noise_prior = _ladder_estimate_noise(stats->sigma, stats->noise_floor, stats->nrungs);

  return TRUE;
}

// implementation-plan-8.md §5.4 Phase 5.1: moved up from just after _mode_shape
// so CT_PICK_PERCENTILE's own case below can call _box_block_percentiles directly
// -- these three functions were already grouped together in the file (plan-8's
// own Phase 3.2 commit put them right after _mode_shape), so the move carries the
// whole group rather than adding a forward declaration (implementation-plan-6.md
// §6 Phase 2.1 set the precedent for preferring this over a forward declaration).
// ascending comparator for the p90/p99 reads below.
static int _ct_cmp_double(const void *a, const void *b)
{
  const double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

// nearest-rank percentile of a sorted (ascending) array of n doubles, frac
// in [0, 1].
static double _ct_percentile_sorted(const double *const restrict sorted, const size_t n,
                                    const double frac)
{
  if(n == 0) return 0.0;
  if(n == 1) return sorted[0];
  const size_t idx = (size_t)(frac * (double)(n - 1) + 0.5);
  return sorted[MIN(idx, n - 1)];
}

// implementation-plan-8.md §5.2/§5.4: percentile mode's own observable
// needs p90/p99 of the per-block RMS a box's blocks carry at each rung --
// the *distribution* §5.4's 4th component (Phase 3.1) publishes, not the
// sum _fit_curve_from_box's SAT query reads. Called by CT_PICK_PERCENTILE's
// own case in _mode_shape below (Phase 5.1).
//
// A super-block's one aggregated RMS value is repeated once per base block
// it covers (the same repetition _ladder_build_blockrms wrote into the
// buffer) rather than de-duplicated here: the percentile then comes out
// implicitly weighted by how much of the box's own *area* each super-block
// covers, which is the physically meaningful weighting for "how
// concentrated is this box's own local contrast" -- a single fine-rung
// super-block sitting in a corner of a large box should not outvote one
// that covers most of it.
//
// nblocks[] gets the number of *distinct* super-blocks the box overlaps at
// each rung -- what the percentile is really a statistic of, since the
// repeated values above add area weight but no new samples. A nearest-rank
// p99 is simply the maximum below ~100 distinct values, and p90 is too
// below 10, so p90/p99 reads exactly 1 on a box that overlaps fewer than
// ~10 super-blocks whatever its content: the caller uses this count to
// know which rungs it can trust (CT_PERCENTILE_MIN_BLOCKS).
//
// The box-to-blocks clipping below is the same arithmetic
// _fit_curve_from_box uses (see there); not factored into a shared helper
// since the two loops that follow it diverge immediately (four-corner SAT
// lookups there, a full block walk here) and there is no third caller yet
// to justify the indirection.
static gboolean _box_block_percentiles(dt_iop_module_t *self, const int *const box,
                                       double *const restrict p90,
                                       double *const restrict p99,
                                       int *const restrict nblocks,
                                       int *const restrict nrungs_out)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);

  const size_t sat_w = g->pd.width, sat_h = g->pd.height;
  const size_t comps = g->pd.components;
  const gboolean have_data =
    g->pd.buf && sat_w > 1 && sat_h > 1 && g->ladder_nrungs > 0
    && comps == (size_t)(4 * g->ladder_nrungs);

  gboolean ok = FALSE;
  if(have_data)
  {
    const size_t bw = sat_w - 1, bh = sat_h - 1;
    const size_t bx0 = MIN(bw, (size_t)MAX(box[0], 0) / CT_BLOCK);
    size_t bx1 = MIN(bw, (size_t)(MAX(box[2], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(bx1 <= bx0) bx1 = MIN(bw, bx0 + 1);
    const size_t by0 = MIN(bh, (size_t)MAX(box[1], 0) / CT_BLOCK);
    size_t by1 = MIN(bh, (size_t)(MAX(box[3], 0) + CT_BLOCK - 1) / CT_BLOCK);
    if(by1 <= by0) by1 = MIN(bh, by0 + 1);

    // implementation-plan-8.md §5.4: O(blocks) per rung, <= 21600 blocks for
    // the whole frame at CT_BLOCK=8 on a 1440x960 preview -- this runs once
    // per pick on the GUI thread, no need for anything past a plain qsort.
    const size_t nbase = (bx1 - bx0) * (by1 - by0);
    double *const restrict scratch = dt_alloc_align_double(MAX(nbase, (size_t)1));
    if(scratch)
    {
      const float *const restrict buf = g->pd.buf;
      *nrungs_out = g->ladder_nrungs;
      for(int r = 0; r < g->ladder_nrungs; r++)
      {
        size_t n = 0;
        for(size_t by = by0; by < by1; by++)
          for(size_t bx = bx0; bx < bx1; bx++)
            scratch[n++] = (double)buf[(by * sat_w + bx) * comps + 4 * r + 3];

        qsort(scratch, n, sizeof(double), _ct_cmp_double);
        p90[r] = _ct_percentile_sorted(scratch, n, 0.90);
        p99[r] = _ct_percentile_sorted(scratch, n, 0.99);

        // super-blocks sit on the global block grid (_ladder_build_blockrms
        // groups from block 0), so the count is over the grid cells the
        // clipped box touches, partial cells at its edges included
        const double step = exp2((double)(r / CT_SCALES_PER_OCTAVE));
        const size_t grp = _ladder_superblock_side(step);
        nblocks[r] = (int)(((bx1 - 1) / grp - bx0 / grp + 1) * ((by1 - 1) / grp - by0 / grp + 1));
      }
      dt_free_align(scratch);
      ok = TRUE;
    }
  }

  dt_iop_gui_leave_critical_section(self);
  return ok;
}

// implementation-plan-8.md §5.2: Bonnier & Simoncelli's own defaults, kept
// as-is -- Phase 0.2's sweep of gamma over {0.3, 0.5, 0.7} on the twelve
// crops found no value clearly better than 0.5.
#define CT_PERCENTILE_GAMMA 0.5
#define CT_PERCENTILE_EPS 0.01

// the two rails percentile mode's q_r = p90/p99 is read against, the same
// way CT_KAPPA_GAUSSIAN/CT_KAPPA_STRUCT bracket structure mode's kappa.
//
// CT_PERCENTILE_Q_NOISE: what a spatially uniform field reads, i.e. the
// finite-sample scatter of 64-sample block RMS values -- the chi
// distribution with 64 degrees of freedom gives p90/p99 = 0.9198, and a
// simulated nearest-rank read of it stays within 0.92-0.94 for any 20 or
// more distinct super-blocks. A rung at or above this rail is as even as
// noise and contributes nothing. §5.2/§5.4's per-frame "measured
// q_gauss" replaced this: it was taken over the frame's near-Gaussian
// super-blocks, but a super-block spans about one wavelength of its own
// rung and reads near-Gaussian on ~78% of *all* content (see
// _ct_structure_shape's comment), so what it measured was the frame's own
// concentration, and a sub-box rarely beats the frame it is cut from --
// plan-8-evidence/01-phase-4-5-review.txt: "already even" on a quarter of
// 300 px boxes, and every rung with fewer than ~10 super-blocks.
//
// CT_PERCENTILE_Q_STRUCT: the q_r at which a rung counts as fully
// concentrated -- 10th percentile of q_r over 3120 measured (box, rung)
// pairs on the same review set, the mirror of CT_KAPPA_STRUCT's own 90th
// percentile rule. An absolute rail rather than §5.2's peak-normalised
// shape, so a box whose contrast is nearly even everywhere gets a small
// curve, not a full-strength one built from whichever rung was least even.
#define CT_PERCENTILE_Q_NOISE 0.9198
#define CT_PERCENTILE_Q_STRUCT 0.43

// fewest distinct super-blocks a rung needs before its p90/p99 means
// anything -- see _box_block_percentiles: below ~10 the ratio is 1 by
// construction, and 20 is where the simulated iid read settles onto the
// closed form above. Rungs run fine to coarse and the count only ever
// falls with rung, so the measurable rungs are always a prefix.
#define CT_PERCENTILE_MIN_BLOCKS 20


// implementation-plan-8.md §5's shared taper/smoothing rule for both
// adaptive modes: "The three finest nodes are never measured at preview
// scale. Each mode's shape is defined on measured rungs only and tapered
// linearly to 0 over one octave past the finest measured rung." Applied
// here as a one-octave-wide linear ramp in log-sigma, past sigma_r[0] (the
// finest measured rung -- rungs run fine-to-coarse, matching
// _ct_box_stats_t's own convention).
//
// implementation-plan-8.md §5: "each mode's per-rung shape is smoothed with
// a one-octave Gaussian in log sigma before it is evaluated on the
// projection grid" -- val_r is smoothed in place, in log-sigma space, with
// a Gaussian kernel of standard deviation one octave. Rungs are evenly
// spaced in log2(sigma) at CT_SCALES_PER_OCTAVE per octave, so this is a
// small discrete convolution over the rung index, weighted by each pair's
// actual log-sigma separation rather than assumed-uniform spacing (true
// near a resolution-boundary octave transition, but cheap enough not to
// assume it).
static void _ct_smooth_log_octave(const double *const restrict sigma_r,
                                  double *const restrict val_r, const int n)
{
  if(n < 2) return;
  double *const restrict smoothed = malloc(sizeof(double) * n);
  if(!smoothed) return;
  for(int i = 0; i < n; i++)
  {
    double wsum = 0.0, vsum = 0.0;
    const double log_si = log2(sigma_r[i]);
    for(int j = 0; j < n; j++)
    {
      const double dz = log2(sigma_r[j]) - log_si;
      const double w = exp(-0.5 * dz * dz);  // sigma = 1 octave
      wsum += w;
      vsum += w * val_r[j];
    }
    smoothed[i] = (wsum > 0.0) ? vsum / wsum : val_r[i];
  }
  memcpy(val_r, smoothed, sizeof(double) * n);
  free(smoothed);
}

// evaluate the per-rung shape (already smoothed, peak normalised to 1) on
// the dense projection grid: log-linear interpolation between measured
// rungs, a one-octave linear taper to 0 past the finest measured rung
// (§5's own rule, see _ct_smooth_log_octave's comment), and constant
// extrapolation (hold the coarsest measured rung's own value) past the
// coarse end -- the plan states a fine-end taper explicitly and says
// nothing about the coarse end, so holding flat there is this
// implementation's own choice, not a stated rule; a future reader who
// wants a coarse-end taper too should treat this as a one-line change, not
// a discovered bug.
static void _ct_project_rung_shape(const double *const restrict sigma_r,
                                   const double *const restrict s_r, const int nrungs,
                                   const double *const restrict sigma_grid, const int m,
                                   double *const restrict shape01)
{
  const double log_finest = log2(sigma_r[0]);
  for(int j = 0; j < m; j++)
  {
    const double log_sg = log2(sigma_grid[j]);
    if(log_sg < log_finest - 1.0) { shape01[j] = 0.0; continue; }
    if(log_sg < log_finest)
    {
      // linear taper, 1 octave wide, from s_r[0] at the finest measured
      // rung down to 0 one octave finer than it.
      shape01[j] = s_r[0] * (1.0 - (log_finest - log_sg));
      continue;
    }
    if(nrungs == 1 || log_sg >= log2(sigma_r[nrungs - 1])) { shape01[j] = s_r[nrungs - 1]; continue; }

    int hi = 1;
    while(hi < nrungs - 1 && log2(sigma_r[hi]) < log_sg) hi++;
    const int lo = hi - 1;
    const double lo_z = log2(sigma_r[lo]), hi_z = log2(sigma_r[hi]);
    const double t = (hi_z > lo_z) ? (log_sg - lo_z) / (hi_z - lo_z) : 0.0;
    shape01[j] = s_r[lo] + t * (s_r[hi] - s_r[lo]);
  }
}

// implementation-plan-8.md §5.2 Phase 5.1: CT_PICK_PERCENTILE's own per-rung
// shape, from the box's p90/p99 block-RMS concentration -- Bonnier &
// Simoncelli's per-subband multiplier g(q) = ((1-eps)*q + eps)^(gamma-1)
// collapsed to one gain per rung, read between the CT_PERCENTILE_Q_NOISE
// and CT_PERCENTILE_Q_STRUCT rails:
//
//   s_r = clamp((g(q_r) - g(Q_NOISE)) / (g(Q_STRUCT) - g(Q_NOISE)), 0, 1)
//
// Only rungs with at least CT_PERCENTILE_MIN_BLOCKS distinct super-blocks
// are measured; the shape past the last measured rung holds that rung's
// own value, the same coarse-end rule _ct_project_rung_shape applies past
// the coarsest rung anyway. *nmeasured reports how many rungs that was --
// 0 means the box was too small to read at any rung, which a caller
// should report differently from an all-zero shape. Returns TRUE with
// `shape[]` filled iff at least one measured rung came back non-zero.
static gboolean _ct_percentile_shape(const _ct_box_stats_t *const stats,
                                     const double *const restrict p90,
                                     const double *const restrict p99,
                                     const int *const restrict nblocks, const int box_nrungs,
                                     const double *const restrict sigma_grid, const int m,
                                     double *const restrict shape, double *const restrict q_out,
                                     int *const nmeasured)
{
  // both queries walk the same live ladder this pick's own `stats` came
  // from and share its fine-to-coarse rung order, so index r here is index
  // r there too -- `_box_block_percentiles` just doesn't stop early at the
  // box's own lambda_max the way `_measure_box` does, hence the MIN.
  const int n = MIN(box_nrungs, stats->nrungs);
  int nm = 0;
  while(nm < n && nblocks[nm] >= CT_PERCENTILE_MIN_BLOCKS && p99[nm] > 0.0) nm++;
  *nmeasured = nm;
  if(nm == 0) return FALSE;

  const double g_noise = pow((1.0 - CT_PERCENTILE_EPS) * CT_PERCENTILE_Q_NOISE + CT_PERCENTILE_EPS,
                             CT_PERCENTILE_GAMMA - 1.0);
  const double g_struct = pow((1.0 - CT_PERCENTILE_EPS) * CT_PERCENTILE_Q_STRUCT + CT_PERCENTILE_EPS,
                              CT_PERCENTILE_GAMMA - 1.0);
  double s_r[CT_MAX_BANDS];
  gboolean any_nonzero = FALSE;
  for(int r = 0; r < nm; r++)
  {
    const double q = fmin(1.0, p90[r] / p99[r]);
    q_out[r] = q;
    const double g_r = pow((1.0 - CT_PERCENTILE_EPS) * q + CT_PERCENTILE_EPS, CT_PERCENTILE_GAMMA - 1.0);
    s_r[r] = CLAMP((g_r - g_noise) / (g_struct - g_noise), 0.0, 1.0);
    if(s_r[r] > 0.0) any_nonzero = TRUE;
  }
  if(!any_nonzero) return FALSE;

  _ct_smooth_log_octave(stats->sigma, s_r, nm);

  double shape01[CT_PROJECT_GRID];
  _ct_project_rung_shape(stats->sigma, s_r, nm, sigma_grid, m, shape01);
  for(int j = 0; j < m; j++) shape[j] = 1.0 + CT_DEFAULT_PEAK_EFF * shape01[j];
  return TRUE;
}

// implementation-plan-8.md §5.1 Phase 4.1: CT_PICK_STRUCTURE's own per-rung
// shape, from the box-wide kappa = sqrt(E_r) / (S1_r / n_eff), smoothed and
// projected onto `sigma_grid` through the shared §5 rule above. Returns
// TRUE with `shape[]` filled iff at least one rung came back non-zero --
// an all-zero result is Phase 4.4's own "nothing to do" case, left to the
// caller to log and refuse.
//
// Box-wide, not the block-median kappa Phase 4.3 asked for. That variant
// was built and measured against the same box picks a user makes
// (plan-8-evidence/01-phase-4-5-review.txt): it came back all-zero on two
// thirds of 300 px and 500 px boxes, because a super-block of ~64 level
// pixels spans about one wavelength of its own rung, and within one
// wavelength a band-pass response is a smooth lobe whose L2/L1 sits at or
// below the Gaussian value whatever the content (pooled per-super-block
// kappa: median 1.20, 90th percentile 1.24, against CT_KAPPA_GAUSSIAN =
// 1.25). Sparseness at a rung's own scale lives in how energy is spread
// *between* super-blocks, which is exactly what the box-wide ratio keeps
// and a per-block median discards. Phase 0.1's "confound" (a box straddling
// two regions reads sparse at the busier region's scales) is that same
// between-block spread, and boosting those rungs boosts the busier region,
// which is the pick the user made.
static gboolean _ct_structure_shape(const _ct_box_stats_t *const stats,
                                    const _ct_fit_t *const fit,
                                    const double *const restrict sigma_grid, const int m,
                                    double *const restrict shape)
{
  const int nrungs = stats->nrungs;
  if(nrungs <= 0) return FALSE;

  // §5.1: s_r = clamp((kappa - CT_KAPPA_GAUSSIAN) / (CT_KAPPA_STRUCT -
  // CT_KAPPA_GAUSSIAN), 0, 1), "S/(S+N) from the fit still multiplied in
  // for the fine rungs" -- residual non-Gaussian noise (demosaic pattern,
  // compression) at the fine end would otherwise read as structure.
  double s_r[CT_MAX_BANDS];
  gboolean any_nonzero = FALSE;
  for(int r = 0; r < nrungs; r++)
  {
    const double kappa = (stats->s1_energy[r] > 0.0) ? sqrt(stats->energies[r]) / stats->s1_energy[r] : 0.0;
    double s = CLAMP((kappa - CT_KAPPA_GAUSSIAN) / (CT_KAPPA_STRUCT - CT_KAPPA_GAUSSIAN), 0.0, 1.0);
    double S, N;
    _ct_fit_eval(fit, stats->sigma[r], &S, &N);
    s *= S / fmax(S + N, DBL_MIN);
    s_r[r] = s;
    if(s > 0.0) any_nonzero = TRUE;
  }
  if(!any_nonzero) return FALSE;

  _ct_smooth_log_octave(stats->sigma, s_r, nrungs);

  double shape01[CT_PROJECT_GRID];
  _ct_project_rung_shape(stats->sigma, s_r, nrungs, sigma_grid, m, shape01);
  for(int j = 0; j < m; j++) shape[j] = 1.0 + CT_DEFAULT_PEAK_EFF * shape01[j];
  return TRUE;
}

// implementation-plan-8.md §5: given a `_measure_box` + `_fit_spectrum` that
// already succeeded, decide what shape `mode` writes onto the projection
// grid `sigma_grid`/`m`. `*absolute` reports whether `shape[]` is already
// the module's own absolute target -- master-independent, the way
// CT_TARGET_DEFAULT/EQUALIZE are in `_target_curve` -- rather than a [0,1]
// shape meant to be lerped with gain_local_contrast; every mode built by
// plan-8 is the former (§5's "the picker sets shape, never strength"), but
// the caller still needs to be told which.
//
// CT_PICK_STRUCTURE (§5.1, `_ct_structure_shape`) and CT_PICK_PERCENTILE
// (§5.2/§5.4, below) each have their own case. Returns FALSE only for a
// mode whose shape comes back all-zero (Phase 4/5's own "nothing to do"
// case -- the default curve is already applied, so this is a no-op, not a
// fallback write); CT_PICK_FIXED's Gaussian hump is never all-zero and so
// never returns FALSE.
//
// `self`/`box` are needed only by CT_PICK_PERCENTILE (§5.4's box + frame
// block-RMS percentile queries read `self->gui_data` directly, and the box
// query needs the original pixel box, not anything `_ct_box_stats_t` keeps)
// -- CT_PICK_FIXED and CT_PICK_STRUCTURE (kappa is already in `stats`) have
// no use for either.
//
// CT_PICK_FIXED's body below is the pre-plan-8 `_fit_curve_from_box`'s own
// post-`_fit_spectrum` code, moved verbatim (the found_texture advisories,
// then the CT_TARGET_DEFAULT hump via `_target_curve`) -- implementation-
// plan-8.md §6 Phase 2.2's own regression bar: "a fixed-mode pick writes
// byte-identical band[] to before."
static gboolean _mode_shape(dt_iop_module_t *self, const int *const box,
                            const _ct_picker_mode_t mode,
                            const _ct_box_stats_t *const stats,
                            const _ct_fit_t *const fit,
                            const double *const restrict sigma_grid, const int m,
                            const double sigma_ref, const float scale_shift,
                            double *const restrict shape,
                            gboolean *const absolute)
{
  // implementation-plan-6.md §5B.2/§6 Phase 2.2: the picker's shape is not
  // derived from what was measured here (§3.7b) for the fixed mode -- kept,
  // for every mode, purely to gate the size-sanity advisories that follow
  // (fit->tau/fit->texture are meaningless without a texture to have
  // measured); §8.2's own note, "fit->tau/fit->texture feed _ct_fit_eval's
  // S(sigma) the same way regardless of mode", is why these advisories run
  // ahead of the mode switch instead of once per case.
  const gboolean found_texture = fit->texture_peak > stats->peak_e * 1e-2;

  // §2.4/research.md §5.9: advisory only, neither warning below refuses the
  // pick -- both just explain a result that might otherwise look like
  // nothing happened, or like an untrustworthy size.
  {
    int peak_idx = 0;
    for(int r = 1; r < stats->nrungs; r++)
      if(stats->energies[r] > stats->energies[peak_idx]) peak_idx = r;
    double S, N;
    _ct_fit_eval(fit, stats->sigma[peak_idx], &S, &N);
    if(S <= (S + N) * CT_NOISE_DOMINATED_FRAC)
      dt_control_log(_("the picked area looks like noise -- try raising the noise bias"));
  }

  if(found_texture)
  {
    const double target_sigma = sqrt(fit->tau);

    // §6.2: the fitted size sits within half an octave of the window's own
    // coarse edge (§1.1's lambda_max) -- there is no peak inside what the
    // box could see, only a rising flank, so the reported size is read off
    // the edge of the window rather than measured. Warn, don't refuse: this
    // is the honest answer, not a bad one.
    if(target_sigma >= stats->sigma[stats->nrungs - 1] / M_SQRT2)
      dt_control_log(_("the measured size sits at the edge of what this box can see -- "
                        "it may be larger than reported"));

    // implementation-plan-3.md §7 (Issue 2d): the mirror-image failure --
    // the box is too small to *contain* the feature, so the fit can't place
    // any peak inside what it measured and instead collapses tau toward the
    // ladder's finest rung, railing beta high to explain the rest. A
    // full-octave margin catches this without false-positiving on a
    // legitimate fine-texture pick -- see findings.md for the sweep this
    // threshold came from.
    if(target_sigma <= stats->sigma[0] * 2.0)
    {
      const double min_side = 2.0 * stats->window_lambda_max;
      dt_control_log(_("the box is too small to see how big this is -- "
                        "try at least %.0f x %.0f px"), min_side, min_side);
    }

    int nearest = 0;
    double best_d = DBL_MAX;
    for(int r = 0; r < stats->nrungs; r++)
    {
      const double dist = fabs(log(stats->sigma[r] / target_sigma));
      if(dist < best_d) { best_d = dist; nearest = r; }
    }
    if(stats->s1_energy[nearest] > 0.0)
    {
      const double kappa = sqrt(stats->energies[nearest]) / stats->s1_energy[nearest];
      if(kappa > CT_KAPPA_EDGE)
        dt_control_log(_("the picked area looks more like a hard edge than dense texture -- "
                          "the measured size may be unreliable"));
    }
  }

  switch(mode)
  {
    // implementation-plan-8.md §5.2/§5.4 Phase 5.1/5.3: spatial
    // concentration per rung -- see `_ct_percentile_shape`.
    case CT_PICK_PERCENTILE:
    {
      double p90[CT_MAX_BANDS], p99[CT_MAX_BANDS];
      int nblocks[CT_MAX_BANDS];
      int box_nrungs = 0;
      if(!_box_block_percentiles(self, box, p90, p99, nblocks, &box_nrungs)) box_nrungs = 0;

      double q_r[CT_MAX_BANDS];
      int nmeasured = 0;
      const gboolean ok = _ct_percentile_shape(stats, p90, p99, nblocks, box_nrungs, sigma_grid, m,
                                               shape, q_r, &nmeasured);

      // implementation-plan-8.md §4.4/§5.4 Phase 2.3/5.2: publish this
      // pick's own q_r for the graph overlay, measured rungs only, whether
      // or not there was anything to write -- the overlay's whole point is
      // to make a pick legible, including one that landed on nothing.
      {
        dt_iop_contrast_gui_data_t *const g = self->gui_data;
        dt_iop_gui_enter_critical_section(self);
        if(nmeasured > 0) memcpy(g->mode_overlay, q_r, sizeof(double) * nmeasured);
        g->mode_overlay_n = nmeasured;
        dt_iop_gui_leave_critical_section(self);
      }

      if(!ok)
      {
        // §5's own "nothing to do" case: the default curve is already
        // applied, so refusing here is a no-op, not a fallback. A box with
        // no measurable rung at all is a different refusal: nothing was
        // read, so nothing can be said about its evenness. _fit_spectrum's
        // own CT_MIN_SPAN refusal does not cover it -- a 16 px box keeps
        // the four rungs the fit wants but overlaps only 2x2 finest-rung
        // super-blocks -- so quote the smallest box that would, the way
        // the fit's own refusal does.
        if(nmeasured == 0)
        {
          const double min_side = CT_BLOCK * ceil(sqrt((double)CT_PERCENTILE_MIN_BLOCKS));
          dt_control_log(_("the picked area is too small to measure local contrast levels from -- "
                            "try at least %.0f x %.0f px"), min_side, min_side);
        }
        else
          dt_control_log(_("local contrast is already even at every size in the picked area"));
        return FALSE;
      }

      *absolute = TRUE;
      return TRUE;
    }

    // implementation-plan-8.md §5.1 Phase 4.1/4.4: structure's own shape,
    // from the box-wide kappa -- see `_ct_structure_shape`.
    case CT_PICK_STRUCTURE:
    {
      if(!_ct_structure_shape(stats, fit, sigma_grid, m, shape))
      {
        // §5's own rule: "a mode whose shape comes back all-zero logs its
        // own message and writes nothing -- the default curve is already
        // applied, so 'nothing to do' is a no-op, not a fallback write."
        dt_control_log(_("nothing in the picked area reads as structure -- "
                          "it is texture or noise at every size"));
        return FALSE;
      }
      *absolute = TRUE;
      return TRUE;
    }

    case CT_PICK_FIXED:
    default:
    {
      _target_curve(fit, CT_TARGET_DEFAULT, sigma_grid, m, sigma_ref, scale_shift, shape);
      *absolute = TRUE;
      return TRUE;
    }
  }
}

// ---------------------------------------------------------------------------
// §3.4: presets, built the same way the picker projects a target curve onto
// the nine bands (_project_to_bands) -- research.md §5.7's own "the same
// solve also gives you preset-building for free" -- but from an analytic
// shape instead of a picked area's fit. Shapes below are hand-picked
// starting points, the same way atrous.c's own shipped presets are plain
// numbers with no cited derivation; nothing here claims the empirical
// grounding phase 0's decomposition choices had.
// ---------------------------------------------------------------------------

// implementation-plan-2.md §5.2: §4.1's frame-relative formula -- H_k
// depends only on the *ratio* between sigma and lambda, so the arbitrary
// large "frame size" this used to need (to keep modify_roi_in's own "-1"
// pixel-discretisation term negligible) was never necessary, and neither
// was the term itself. scale_shift = 0: presets sit on the standard ladder.
//
// project target[] (evaluated on lambda_grid[]) onto the nine bands and
// write the result into p->band[], coarsest-first -- the same reversal
// color_picker_apply does at the end of its own projection. gain_lo/gain_hi
// (§4.4) is this preset's own envelope: most presets built target[] as
// 1 + strength*shape(), so their envelope is exactly [1, 1+strength] (or
// the reverse for a sub-1 preset like "soften"); "flatten spectrum" already
// clamps target[] itself, so its own bounds are what it passes here.
static gboolean _preset_apply_target(const double *const restrict lambda_grid,
                                     const double *const restrict target, const int m,
                                     const float *const restrict sigma,
                                     const float gain_lo, const float gain_hi,
                                     dt_iop_contrast_params_t *const p)
{
  float gains[CT_BANDS];
  if(!_project_to_bands(lambda_grid, target, m, sigma, CT_BANDS, NULL, gain_lo, gain_hi, gains))
    return FALSE;
  for(int k = 0; k < CT_BANDS; k++) p->band[k] = gains[CT_BANDS - 1 - k];
  return TRUE;
}

// raised-cosine bump on x in [0,1] (the graph's own coarse=0/fine=1 axis,
// via _spectrum_lambda_to_x), centred at x0 with half-width w -- zero
// outside [x0-w, x0+w], peak 1 at x0.
static double _preset_bump(const double x, const double x0, const double w)
{
  const double t = CLAMP((x - x0) / w, -1.0, 1.0);
  return 0.5 * (1.0 + cos(M_PI * t));
}

// implementation-plan-8.md §3.1: the module's default curve is the fixed
// hump (CT_TARGET_DEFAULT, §5B.1) projected onto band[] through the exact
// same path init_presets' "default curve" preset (below) and a fixed-mode
// pick (_color_picker_apply_now) use -- not nine hardcoded numbers of its
// own, so all three can never disagree by rounding. scale_shift = 0, no
// calibration, gain envelope [1, 1+CT_DEFAULT_PEAK_EFF] mirrors exactly
// what _color_picker_apply_now computes for mode == CT_TARGET_DEFAULT.
void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);

  dt_iop_contrast_params_t *const d = self->default_params;

  float sigma[CT_BANDS];
  _ct_band_sigma(sigma, 0.0f);
  double lo, hi;
  _ct_grid_bounds(sigma, &lo, &hi);

  double lambda_grid[CT_PROJECT_GRID], sigma_grid[CT_PROJECT_GRID], shape[CT_PROJECT_GRID];
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    sigma_grid[j] = lo * exp2(log2(hi / lo) * (double)j / (double)(CT_PROJECT_GRID - 1));
    lambda_grid[j] = sigma_grid[j] * CT_SIGMA_TO_LAMBDA;
  }

  // _target_curve's CT_TARGET_DEFAULT branch reads neither fit nor
  // sigma_ref (only CT_TARGET_EQUALIZE does) -- a zeroed fit and a
  // sigma_ref of 0 are safe here.
  _ct_fit_t unused_fit;
  memset(&unused_fit, 0, sizeof(unused_fit));
  _target_curve(&unused_fit, CT_TARGET_DEFAULT, sigma_grid, CT_PROJECT_GRID, 0.0, 0.0f, shape);

  _preset_apply_target(lambda_grid, shape, CT_PROJECT_GRID, sigma, 1.0f, 1.0f + CT_DEFAULT_PEAK_EFF, d);
}

void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_contrast_params_t p;
  memset(&p, 0, sizeof(p));
  p.gain_local_contrast = 1.0f;
  p.scale_shift = 0.0f;
  p.edge_protection = 0.0f;
  p.filter_iterations = 1;
  p.noise_bias = 0.001f;
  for(int k = 0; k < CT_BANDS; k++) p.band[k] = 1.0f;

  // implementation-plan-2.md §5.2/§4.3: sigma-native grid, frame-relative
  // (long edge = 1.0 throughout -- _spectrum_lambda_to_x's roi_long_edge
  // argument, and _ct_band_sigma above with scale_shift = 0), matching the
  // picker's own §4.3 grid exactly -- implementation-plan-3.md §3.1's coarse
  // trim (see color_picker_apply's own grid comment) included, since §3.3
  // depends on the two staying identical.
  float sigma[CT_BANDS];
  _ct_band_sigma(sigma, 0.0f);
  double lambda_grid[CT_PROJECT_GRID], sigma_grid[CT_PROJECT_GRID], target[CT_PROJECT_GRID];
  double lo, hi;
  _ct_grid_bounds(sigma, &lo, &hi);
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    sigma_grid[j] = lo * exp2(log2(hi / lo) * (double)j / (double)(CT_PROJECT_GRID - 1));
    lambda_grid[j] = sigma_grid[j] * CT_SIGMA_TO_LAMBDA;
  }

  // "default curve" (plan-8 §3.3): the module's own baseline (§3.1's
  // init()) as an explicit preset, so it is reachable again after a pick or
  // a hand edit without resetting the whole module, and so the preset list
  // states the baseline next to clarity/texture/etc rather than leaving it
  // implicit. Same projection as init() -- CT_TARGET_DEFAULT's shape at
  // scale_shift = 0, envelope [1, 1+CT_DEFAULT_PEAK_EFF] -- not a
  // hand-rolled bump like the presets below it.
  {
    _ct_fit_t unused_fit;
    memset(&unused_fit, 0, sizeof(unused_fit));
    _target_curve(&unused_fit, CT_TARGET_DEFAULT, sigma_grid, CT_PROJECT_GRID, 0.0, 0.0f, target);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.0f + CT_DEFAULT_PEAK_EFF, &p))
    dt_gui_presets_add_generic(_("default curve"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "clarity": a broad boost centred mid-ladder, slightly toward the coarse
  // side -- traditional medium/large-scale local contrast.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    target[j] = 1.0 + 0.6 * _preset_bump(x, 0.35, 0.5);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.6f, &p))
    dt_gui_presets_add_generic(_("clarity"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "texture": a broader boost biased toward the fine end -- less peaked
  // than micro-contrast below, meant as a general "add texture" default.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    target[j] = 1.0 + 0.5 * _preset_bump(x, 0.7, 0.6);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.5f, &p))
    dt_gui_presets_add_generic(_("texture"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "micro-contrast": only the finest couple of bands, ramped in rather
  // than a bump, for a tighter, more surgical fine-detail boost.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    const double ramp = pow(CLAMP((x - 0.55) / 0.45, 0.0, 1.0), 1.5);
    target[j] = 1.0 + 0.8 * ramp;
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 1.0f, 1.8f, &p))
    dt_gui_presets_add_generic(_("micro-contrast"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "soften": the inverse of micro-contrast -- an edge-aware fine-detail
  // smoother, g_k < 1 toward the finest bands, untouched at the coarse end.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    const double x = _spectrum_lambda_to_x(lambda_grid[j], 1.0);
    const double ramp = CLAMP((x - 0.5) / 0.5, 0.0, 1.0);
    target[j] = 1.0 - 0.7 * ramp;
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, 0.3f, 1.0f, &p))
    dt_gui_presets_add_generic(_("soften"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);

  // "flatten spectrum": research.md §5.6's "equalize" mode, now also the
  // live picker's CT_TARGET_EQUALIZE (§8.1) -- same _target_curve, evaluated
  // against a synthetic self-similar spectrum (no sized texture) rather than
  // any particular picked area's own fit, since a preset has no box to
  // measure.
  //
  // Both constants below were re-derived (findings.md, "flatten spectrum"
  // preset section) after Phase 7.3/§5.2 flagged the old beta = 2.4 as
  // research.md §5.3's literature figure for *linear-radiance* power spectra
  // applied to a ladder that measures log2 luminance -- never verified in
  // this domain. beta = 2.75 is the median of Phase 2's refined-beta fits
  // across twelve real log2-luminance crops (implementation-plan-3.md §2,
  // 1.8-3.4 cluster). noise = 1.8e-10 replaces the old 0.02: measured
  // directly (six real crops, self_similar/noise both fitted together) as
  // the median noise/self_similar ratio at this struct's self_similar = 1.0
  // reference scale -- the old value was ~8 orders of magnitude too large at
  // that scale, which silently pinned wiener = S/(S+N) near 0 at every band
  // but the coarsest two or three regardless of beta, so the preset's
  // projected gains sat at the CT_EQUALIZE_GAIN_LO floor on nearly every
  // band no matter what beta was set to -- the beta mismatch above was real
  // but had no visible effect until this was fixed too.
  {
    const _ct_fit_t synthetic = { .self_similar = 1.0, .beta = 2.75, .noise = 1.8e-10,
                                  .texture = 0.0, .tau = 0.0 };
    // implementation-plan-3.md §5.1: sigma_ref is the band ladder's own
    // geometric mean, not the grid's -- see _target_curve's comment.
    const double sigma_ref = sqrt((double)sigma[0] * (double)sigma[CT_BANDS - 1]);
    _target_curve(&synthetic, CT_TARGET_EQUALIZE, sigma_grid, CT_PROJECT_GRID, sigma_ref, 0.0f, target);
  }
  if(_preset_apply_target(lambda_grid, target, CT_PROJECT_GRID, sigma, CT_EQUALIZE_GAIN_LO, CT_EQUALIZE_GAIN_HI, &p))
    dt_gui_presets_add_generic(_("flatten spectrum"), self->op, self->version(), &p, sizeof(p), TRUE,
                               DEVELOP_BLEND_CS_RGB_SCENE);
}

// §2.2: synchronous now that the ladder is frame-wide and pre-published
// (§2.1/§2.2 above) -- no more *deliberately* arming a pick and waiting for
// a preview pass to claim and measure it (research.md §5.11). the box maps
// straight from the color picker's sample to the ladder's own roi, the SAT
// query is a handful of lookups, and the fit is a grid search over ~80
// points: all fast enough to run inline on the GUI thread instead of
// round-tripping through another preview pass. color_picker_apply below
// still has to cope with one *involuntary* wait -- g->pd occasionally
// racing the pick itself stale, see there -- but that path is the
// exception, not the normal one this function serves.
//
// does the actual fit-and-apply once g->pd is known fresh -- factored out of
// color_picker_apply so the same work can be retried later, off a signal
// rather than off the picker callback itself (see there).
static void _color_picker_apply_now(dt_iop_module_t *self,
                                    dt_dev_pixelpipe_t *pipe)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);
  const dt_iop_roi_t roi_in = g->ladder_roi_in;
  dt_iop_gui_leave_critical_section(self);

  // implementation-plan-2.md §4.2: the ladder's sigma is in the pixels of
  // this same roi_in -- dividing by the roi's own long edge (the same
  // quantity modify_roi_in computes as S * roi_in->scale) is what makes the
  // fit's tau frame-relative, and so comparable to §4.1's frame-relative
  // band sigma below.
  const float long_edge = (float)(MAX(pipe->iwidth, pipe->iheight)) * (float)roi_in.scale;

  // region defaults to the whole frame; a box pick narrows it, same
  // fallback picked-region measurement always used.
  int box[4] = { 0, 0, (int)roi_in.width, (int)roi_in.height };
  const dt_colorpicker_sample_t *const sample =
    darktable.lib->proxy.colorpicker.primary_sample;
  if(sample && sample->size == DT_LIB_COLORPICKER_SIZE_BOX)
  {
    int picked[4];
    if(!dt_color_picker_box(self, &roi_in, sample, PIXELPIPE_PICKER_INPUT, picked))
      memcpy(box, picked, sizeof(box));
  }

  // implementation-plan-8.md §5's shared skeleton: _measure_box, then
  // _fit_spectrum, then a mode-specific _mode_shape, then _project_to_bands
  // (below). Every picker mode shares the first two steps and their
  // refusal messages -- the refusals are properties of the box, not of the
  // mode (§5's own "every mode still runs _fit_spectrum first").
  _ct_box_stats_t stats;
  if(!_measure_box(self, box, long_edge, &stats))
  {
    dt_control_log(_("the preview isn't ready to measure yet -- try again in a moment"));
    return;
  }

  _ct_fit_t fit;
  _ct_fit_refusal_t refusal = CT_FIT_REFUSED_FLAT;
  if(!_fit_spectrum(stats.sigma, stats.energies, stats.weights, stats.nrungs, stats.noise_prior,
                    &fit, &refusal))
  {
    // §6.1: say *which* refusal this is instead of one message covering
    // both -- "too small" and "flat" want different reactions from the
    // user.
    if(refusal == CT_FIT_REFUSED_SPAN)
    {
      // the smallest box that would work at the current preview scale is
      // computable: CT_MIN_SPAN octaves' worth of the finest rung's own
      // wavelength -- the loosest lower bound §1.1's window allows.
      const double min_side = CT_MIN_SPAN * stats.ladder_lambda0;
      dt_control_log(_("the picked area is too small to measure a detail size from -- "
                        "try at least %.0f x %.0f px"), min_side, min_side);
    }
    else
    {
      dt_control_log(_("the picked area has nothing to measure a detail size from -- "
                        "flat sky, a blown highlight and a black frame all look like this"));
    }
    return;
  }

  // §3.2: publish this pick's own spectrum + fit for the graph's live
  // overlay -- a record of the last measurement, independent of whether the
  // picker itself is still "fresh" by the time it gets drawn.
  dt_iop_gui_enter_critical_section(self);
  memcpy(g->spectrum_lambda, stats.lambda, sizeof(double) * stats.nrungs);
  memcpy(g->spectrum_energy, stats.energies, sizeof(double) * stats.nrungs);
  g->spectrum_nrungs = stats.nrungs;
  g->spectrum_noise = fit.noise;
  g->spectrum_self_similar = fit.self_similar;
  g->spectrum_texture = fit.texture;
  g->spectrum_tau = fit.tau;
  g->spectrum_beta = fit.beta;
  g->spectrum_valid = TRUE;
  dt_iop_gui_leave_critical_section(self);

  // write into params and commit *before* refreshing the widget: the refresh
  // below runs under the gui-update guard and so deliberately writes nothing
  // back, which is the whole point of that guard -- it normally runs the other
  // way around, syncing widgets to params that have already changed.
  dt_iop_contrast_params_t *p = self->params;

  // plan-8 §3.2 Phase 1.2: the picker never touches gain_local_contrast any
  // more -- CT_TARGET_DEFAULT's shape now carries its own full effective
  // strength directly (CT_DEFAULT_PEAK_EFF, written at whatever master is
  // already set), so there is no "shape with no strength behind it" case
  // left to rescue the way blackwhite's picker still has to for its own
  // filter-enable exception. The graph at gain_local_contrast == 1.0 now
  // shows exactly what a fixed-mode pick applies -- plan-7 §4.2's "the
  // graph tells the truth", finished.
  //
  // implementation-plan-2.md §4.1: the nominal per-band boundary sigma,
  // frame-relative (sigma / long edge) and finest-first (idx 0) to match
  // `_project_to_bands`'s H_k derivation. The band ladder is frame-relative
  // by construction (node k's nominal wavelength is S * 2^-(D0+k+shift)) --
  // the "-1" pixel-discretisation term and the roi_in.scale factor belong to
  // modify_roi_in, where a sigma has to come out in the pixels of an actual
  // buffer; the projection has no such need, and at preview scale the old
  // formula put sigma[0] at exactly 0 (a zero H_0 column, the finest band
  // permanently unreachable from a pick) and spanned 21 octaves, most of it
  // sub-pixel garbage.
  float sigma[CT_BANDS];
  _ct_band_sigma(sigma, p->scale_shift);

  // implementation-plan-2.md §4.3: dense log-sigma grid spanning the node
  // ladder itself, padded two octaves at the fine end (sigma[0]*0.25, none
  // of it below the finest band) so the projection sees that end band's
  // full response rather than a truncated one. _target_curve/_ct_fit_eval
  // are evaluated directly on sigma_grid; _project_to_bands' H_k needs a
  // real wavelength, so lambda_grid is sigma_grid scaled by
  // CT_SIGMA_TO_LAMBDA.
  //
  // implementation-plan-3.md §3.1: the fine padding above is real -- band 0
  // is a shelf (HP_0 relative to sigma = 0), so sum_k H_k is exactly 1 out
  // to the grid's fine end and every one of those rows is answerable. The
  // coarse padding was not symmetric with it: the basis telescopes to
  // HP_{n-1}, which decays to zero, and sigma[CT_BANDS-1]*4 put the grid's
  // coarse end at lambda = 1.771 long edges -- 2.62 octaves past the
  // coarsest band's own peak response (CT_BAND_PEAK_FACTOR*sigma[7] =
  // 0.2889 long edges) and outside the frame entirely, where nothing was
  // measured and nothing can be applied. Stop at the long edge instead;
  // §3.2 below weights the in-frame remainder that the ladder can still
  // only partly reach.
  //
  // Not the windowing implementation-plan-3.md's Issue 2 (f) measured and
  // rejected: that one moved the grid per pick, with the box, which is what
  // moved s_ref (§5.1) and let the outermost band absorb the tail. This
  // bound is fixed, frame-relative, and identical for every pick and every
  // preset.
  double lambda_grid[CT_PROJECT_GRID], sigma_grid[CT_PROJECT_GRID];
  double shape[CT_PROJECT_GRID], target[CT_PROJECT_GRID];
  double lo, hi;
  _ct_grid_bounds(sigma, &lo, &hi);
  for(int j = 0; j < CT_PROJECT_GRID; j++)
  {
    sigma_grid[j] = lo * exp2(log2(hi / lo) * (double)j / (double)(CT_PROJECT_GRID - 1));
    lambda_grid[j] = sigma_grid[j] * CT_SIGMA_TO_LAMBDA;
  }

  // implementation-plan-3.md §5.1: sigma_ref is the band ladder's own
  // geometric mean, not the grid's -- see _target_curve's comment.
  const double sigma_ref = sqrt((double)sigma[0] * (double)sigma[CT_BANDS - 1]);
  // implementation-plan-8.md §5/§4.2: which shape this pick writes is now
  // the dropbox's own choice -- read straight off the combobox, since it's
  // a GUI preference with no cached copy to go stale (see its own comment
  // in gui_init).
  const _ct_picker_mode_t picker_mode = (_ct_picker_mode_t)dt_bauhaus_combobox_get(g->picker_mode);

  // implementation-plan-8.md §4.4/§5.4 Phase 4.2: CT_PICK_STRUCTURE's own
  // overlay diagnostic -- the box-wide kappa per rung, the same figure
  // `_ct_structure_shape` builds the curve from. Published
  // unconditionally, even if `_mode_shape` below finds nothing to write,
  // since the overlay's whole point is to make a pick legible -- including
  // a pick that landed on nothing.
  dt_iop_gui_enter_critical_section(self);
  if(picker_mode == CT_PICK_STRUCTURE)
  {
    for(int r = 0; r < stats.nrungs; r++)
      g->mode_overlay[r] = (stats.s1_energy[r] > 0.0) ? sqrt(stats.energies[r]) / stats.s1_energy[r] : 0.0;
    g->mode_overlay_n = stats.nrungs;
  }
  else
  {
    g->mode_overlay_n = 0;
  }
  dt_iop_gui_leave_critical_section(self);

  gboolean is_absolute_target = FALSE;
  if(!_mode_shape(self, box, picker_mode, &stats, &fit, sigma_grid, CT_PROJECT_GRID, sigma_ref,
                  p->scale_shift, shape, &is_absolute_target))
    return;  // Phase 4/5's all-zero case: the default curve is already applied, nothing to do

  // §8.1: an absolute target (every mode plan-8 builds, §5's "the picker
  // sets shape, never strength") is used as target[] as-is, not lerped
  // between 1 and master the way a relative [0,1] shape would be --
  // master-independent by design, so a pick lands at the same effective
  // strength regardless of what the master slider was set to beforehand.
  for(int j = 0; j < CT_PROJECT_GRID; j++)
    target[j] = is_absolute_target ? shape[j] : 1.0 + ((double)p->gain_local_contrast - 1.0) * shape[j];

  // §3.1: how much of its own linear H_k each band actually delivered over
  // this same box, last time the module's own bands were measured there --
  // uncalibrated (all 1s) if that measurement isn't available yet.
  float calibration[CT_BANDS];
  _compute_band_calibration(self, box, long_edge, &fit, calibration);

  // §4.4: the envelope the target curve itself was built to. No band should
  // leave it. implementation-plan-6.md §5.3/§6 Phase 2.3/implementation-
  // plan-8.md §5: every picker-reachable target is now absolute, so this is
  // [1, 1+CT_DEFAULT_PEAK_EFF] unconditionally -- gain_lo pinned at 1.0 (not
  // min(1,master)) so R1 (never cut a band below neutral on the autopick
  // path) holds by construction, and gain_hi at the shape's own known
  // ceiling (the Gaussian never exceeds 1) rather than anything
  // master-derived, since this target is master-independent. The
  // CT_TARGET_EQUALIZE bounds this ternary used to carry were already dead
  // on the picker path -- _mode_shape never produces that mode -- and are
  // dropped rather than threaded through as a case nothing can reach.
  const float gain_lo = is_absolute_target ? 1.0f : fminf(1.0f, p->gain_local_contrast);
  const float gain_hi = is_absolute_target ? (1.0f + CT_DEFAULT_PEAK_EFF) : fmaxf(1.0f, p->gain_local_contrast);

  float gains[CT_BANDS];  // finest-first, matching sigma[] above
  if(!_project_to_bands(lambda_grid, target, CT_PROJECT_GRID, sigma, CT_BANDS, calibration,
                        gain_lo, gain_hi, gains))
  {
    dt_control_log(_("could not fit a curve to the picked area"));
    return;
  }

  for(int k = 0; k < CT_BANDS; k++)
    p->band[k] = gains[CT_BANDS - 1 - k];  // back to coarsest-first param order

  dt_dev_add_history_item(darktable.develop, self, TRUE);

  // pull the bound widgets back from params by hand rather than through
  // dt_iop_gui_update(): that also refreshes the blending UI, which switches
  // the color picker off outright (keep = FALSE) whenever blending is not in
  // parametric mode -- and the picker that asked for this measurement is meant
  // to stay armed so the area can be dragged again.
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_update_from_field(self, NULL, NULL, NULL);
  DT_LEAVE_GUI_UPDATE();
}

void color_picker_apply(dt_iop_module_t *self,
                        GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  DT_GUARD_GUI_UPDATE();

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(!g || picker != g->scale_shift) return;

  if(!dt_preview_data_is_fresh(&g->pd))
  {
    // DT_SIGNAL_CONTROL_PICKERDATA_READY (gui/color_picker_proxy.c) is
    // dispatched through a GLib idle source rather than run synchronously
    // on the pipe thread that raised it, so by the time this runs on the
    // GUI thread the live pipe g->pd is checked against can already have
    // moved past the pass that filled g->pd -- a race, not a "the preview
    // hasn't started yet" case, and it doesn't self-heal: the picker only
    // re-fires this callback when the box itself changes
    // (_record_point_area in that same file), so a pick that lands in this
    // window would otherwise sit dropped until the user re-arms the picker
    // by hand. Remember it and force a fresh preview pass; the pass's own
    // finished signal retries it below.
    g->pick_pending = TRUE;
    dt_control_log(_("wait for the preview to finish recomputing"));
    dt_dev_reprocess_preview(self->dev, self->iop_order);
    return;
  }

  g->pick_pending = FALSE;
  _color_picker_apply_now(self, pipe);
}

// retries a pick color_picker_apply had to defer because g->pd was still
// stale -- see the race described there. Connected to
// DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED in gui_init.
static void _preview_pipe_finished_retry_pick(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(!g || !g->pick_pending) return;

  // the picker may have been disarmed, or handed to a different widget or
  // module, while this pick waited on a fresh pass -- applying data nobody
  // is asking for any more would be worse than the bug this is fixing.
  const dt_iop_color_picker_t *const picker = darktable.lib->proxy.colorpicker.picker_proxy;
  if(!picker || picker->module != self || picker->colorpick != g->scale_shift)
  {
    g->pick_pending = FALSE;
    return;
  }

  if(!dt_preview_data_is_fresh(&g->pd)) return;  // still catching up: wait for the next signal

  g->pick_pending = FALSE;
  _color_picker_apply_now(self, self->dev->preview_pipe);
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  if(in) return;

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(g) g->pick_pending = FALSE;

  dt_iop_color_picker_reset(self, TRUE);
}

// shared by every quad (§1.6): early return if the blend module is already
// displaying a mask -- unchanged from the module's original single-mode
// check, now called from both the slider quads and the graph's ctrl+click.
static gboolean _mask_display_blocked(dt_iop_module_t *self)
{
  if(!self->request_mask_display) return FALSE;
  dt_control_log(_("cannot display masks when the blending mask is displayed"));
  return TRUE;
}

// sets g->details_display (toggling it off if it was already showing mode)
// and keeps every quad's active state in sync with it -- exactly one active
// at a time, matching what process() will actually display next pass.
static void _apply_details_display(dt_iop_module_t *self, const dt_iop_details_display_t mode)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_iop_request_focus(self);
  // Activate the module if it wasn't
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  g->details_display = (g->details_display == mode) ? DT_CT_MASK_OFF : mode;

  dt_bauhaus_widget_set_quad_active
    (GTK_WIDGET(g->gain_local_contrast),
     g->details_display == DT_CT_MASK_CORRECTION || g->details_display == DT_CT_MASK_DETAIL);
  for(int k = 0; k < CT_BANDS; k++)
    dt_bauhaus_widget_set_quad_active(GTK_WIDGET(g->band[k]), g->details_display == k);

  dt_iop_refresh_center(self);
}

// the master gain's quad shows CORRECTION (ctrl+click: DETAIL); each band
// slider's quad shows that band's own raw texture ("ct-band" data set at
// creation, gui_init).
static void show_details_callback(GtkWidget *togglebutton, dt_iop_module_t *self)
{
  if(_mask_display_blocked(self))
  {
    dt_bauhaus_widget_set_quad_active(GTK_WIDGET(togglebutton), FALSE);
    return;
  }

  DT_GUARD_GUI_UPDATE();

  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_iop_details_display_t mode;
  if(togglebutton == g->gain_local_contrast)
    mode = dt_modifier_is(dt_key_modifier_state(), GDK_CONTROL_MASK)
      ? DT_CT_MASK_DETAIL : DT_CT_MASK_CORRECTION;
  else
    mode = (dt_iop_details_display_t)GPOINTER_TO_INT(g_object_get_data(G_OBJECT(togglebutton), "ct-band"));

  _apply_details_display(self, mode);
}

// implementation-plan-8.md §4.1: writes the picker-mode preference to conf
// -- a GUI setting, not a param, so this is the combobox's only side
// effect: no dt_dev_add_history_item, no dirty flag, no pipe reprocess. The
// next pick (color_picker_apply) reads the combobox itself when it needs
// the mode, so there is nothing else to keep in sync here.
static void _picker_mode_callback(GtkWidget *combo, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_conf_set_int(CT_PICKER_MODE_CONF, dt_bauhaus_combobox_get(combo));
}

// ---------------------------------------------------------------------------
// the graph (implementation-plan.md §1.4)
// ---------------------------------------------------------------------------
//
// nodes run coarse (left) to fine (right), evenly spaced in *raw* x -- one
// per octave at (k + 0.5) / CT_BANDS, which is exactly what CT_BANDS is --
// but implementation-plan-3.md §4.2 makes the x axis the projection grid's
// own span rather than [0,1] directly, so a node's *screen* x fraction is
// that raw position run through _graph_axis/_graph_node_x, and the nodes
// occupy only the axis's own middle stretch (the grid's fine/coarse padding
// takes the rest -- see _graph_axis's comment). y is log2 gain (§4.1),
// symmetric about CT_GRAPH_LOG_HALF; a node dragged past the visible top or
// bottom now rides the axis edge exactly, since the axis *is* the slider's
// own hard range [0.2, 5.0] -- see _graph_gain_at.
//
// every handler below re-derives the graph's pixel geometry from the
// widget's current allocation rather than caching it, which is what keeps a
// resize from desyncing the nodes (§1.4 acceptance).

// implementation-plan-3.md §4.1: gain -> the graph's y fraction (0 at the
// bottom, 1 at the top) on the log axis. Shared by node placement and the
// envelope rails below.
static float _graph_gain_to_yfrac(const float gain)
{
  return CLAMP(0.5f + log2f(fmaxf(gain, 1e-6f)) / (2.0f * CT_GRAPH_LOG_HALF), 0.0f, 1.0f);
}

// implementation-plan-6.md §6 Phase 4.1: the node position is the *shape*
// the picker wrote (or the user hand-drew) -- process() then multiplies its
// deviation from 1 by the master gain once more (§2.2's `correction *= gate
// * gain_local_contrast`), so what actually reaches the pixels is this,
// which can go negative (inverting that octave's detail) even on a shape
// that itself never goes below zero.
//
// implementation-plan-7.md §4.2: `master` here is no longer the raw slider
// value directly -- it is that band's own post-knee master
// (_ct_band_master), the same per-band value process() actually multiplies
// by (§4.1(d)/§6 Phase 1.3). Using the raw slider value here, as before this
// plan, is exactly §2.2's bug: past each band's own R_k the dashed line kept
// climbing while the pixels stood still.
static float _graph_effective_gain(const int k, const float band_gain, const float master,
                                   const float scale_shift)
{
  const float band_master = (float)_ct_band_master(k, band_gain, master, scale_shift);
  return 1.0f + band_master * (band_gain - 1.0f);
}

static void _graph_curve_from_params(dt_draw_curve_t *curve,
                                     const dt_iop_contrast_params_t *const p,
                                     const _ct_axis_t *const axis)
{
  for(int k = 0; k < CT_BANDS; k++)
    dt_draw_curve_set_point(curve, k, _graph_node_x(k, axis),
                            _graph_gain_to_yfrac(p->band[k]));
}

static void _graph_geometry(GtkWidget *widget, int *inset, int *width, int *height)
{
  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  *inset = DT_PIXEL_APPLY_DPI(4);
  *width = allocation.width - 2 * (*inset);
  *height = allocation.height - 2 * (*inset) - DT_RESIZE_HANDLE_SIZE;
}

// inverse of _graph_node_x: a screen x fraction back to the band whose
// octave cell it falls in -- implementation-plan-3.md §4.2's axis is an
// affine map of raw x, so this is the same inversion as before, just
// through the axis first.
static int _graph_band_at(const int width, const double x, const _ct_axis_t *const axis)
{
  const double xfrac = x / (double)MAX(width, 1);
  const double raw = axis->x0 + xfrac * (axis->x1 - axis->x0);
  const int k = (int)floor(raw * (double)CT_BANDS);
  return CLAMP(k, 0, CT_BANDS - 1);
}

// inverse of the node-drawing map in _area_draw: pixel y (0 at the graph's
// top) to a gain. implementation-plan-3.md §4.1: the top of the axis *is*
// the slider's own hard range now (5.0, and 1/5.0 at the bottom), so this is
// a plain clamp rather than the open-ended climb the old linear axis needed
// -- a node dragged to the floor lands on 0.2, not 0 (the graph cannot draw
// gain = 0; the slider still reaches it, and double-click still resets to
// 1.0, so nothing is unreachable, just not draggable to that exact edge).
static float _graph_gain_at(const int height, const double y)
{
  const float yfrac = CLAMP(1.0f - (float)(y / (double)MAX(height, 1)), 0.0f, 1.0f);
  return (float)exp2((yfrac - 0.5f) * 2.0f * CT_GRAPH_LOG_HALF);
}

static void _area_set_band(dt_iop_contrast_gui_data_t *g, const int k, const float gain)
{
  if(k < 0 || k >= CT_BANDS || !g->band[k]) return;
  dt_bauhaus_slider_set(g->band[k], gain);
}

// ---------------------------------------------------------------------------
// §3.2: the live spectrum overlay -- research.md's own suggestion (§5.6,
// implementation-plan.md §1.4 step 4/toneequal's inset histogram) to draw
// the frame-wide ladder as the graph's background at all times, and the
// last picked box's own spectrum plus its fitted model on top of it once a
// pick has landed. Makes the picker legible instead of magic: the data was
// already being measured (§2.1/§3.1), this just puts it on screen.
// ---------------------------------------------------------------------------

// energies span orders of magnitude across ordinary images (phase0-band-
// energy.md's own tables), so the y axis here is log, normalised to
// whichever curve on screen has the higher peak -- purely relative shape,
// no fixed absolute meaning.
#define CT_SPECTRUM_LOG_RANGE 10.0  // stops of dynamic range shown below the on-screen peak

static float _spectrum_energy_to_y(const double energy, const double peak)
{
  if(energy <= 0.0 || peak <= 0.0) return 0.0f;
  return CLAMP((float)(1.0 + log2(energy / peak) / CT_SPECTRUM_LOG_RANGE), 0.0f, 1.0f);
}

// implementation-plan-8.md §4.4 Phase 4.2: kappa's own y-mapping, for
// CT_PICK_STRUCTURE's overlay -- linear, not the log/peak-relative mapping
// above: kappa is already a bounded O(1) ratio (>= 1.0 by the L2/L1
// power-mean inequality), so a log-relative-to-peak scale would waste most
// of the plot on values that never occur. The range is fixed, not
// peak-relative, so the CT_KAPPA_GAUSSIAN/CT_KAPPA_STRUCT rails sit at the
// same screen height on every pick rather than sliding around with
// whatever kappa this box happened to reach.
#define CT_KAPPA_PLOT_LO 1.0
#define CT_KAPPA_PLOT_HI (CT_KAPPA_STRUCT + (CT_KAPPA_STRUCT - CT_KAPPA_GAUSSIAN) * 0.3)

static float _kappa_to_y(const double kappa)
{
  return CLAMP((float)((kappa - CT_KAPPA_PLOT_LO) / (CT_KAPPA_PLOT_HI - CT_KAPPA_PLOT_LO)), 0.0f, 1.0f);
}

// the frame-wide ladder's own spectrum: one (lambda, energy) point per rung,
// energy = S2/n over the *whole* ladder grid. Since a summed-area table is
// zero-padded on its low side (§2.1's _ladder_build_sat), the frame total is
// simply the table's own opposite corner -- no subtraction needed, unlike a
// box query.
static gboolean _spectrum_frame_wide(dt_iop_module_t *self,
                                     double *const restrict lambda,
                                     double *const restrict energy,
                                     int *const restrict nrungs)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;
  gboolean ok = FALSE;

  dt_iop_gui_enter_critical_section(self);
  const size_t sat_w = g->pd.width, sat_h = g->pd.height;
  const size_t comps = g->pd.components;
  if(g->pd.buf && sat_w > 1 && sat_h > 1 && g->ladder_nrungs > 0
     && comps == (size_t)(4 * g->ladder_nrungs))
  {
    const float *const restrict buf = g->pd.buf;
    const size_t corner = (sat_h - 1) * sat_w + (sat_w - 1);
    *nrungs = g->ladder_nrungs;
    for(int r = 0; r < g->ladder_nrungs; r++)
    {
      // implementation-plan-4.md §8.2: n_eff is sat_n's own corner -- the
      // real level-pixel count the ladder actually summed, not
      // nblocks*(CT_BLOCK/step)^2's assumption that every block (including
      // the grid's own clipped last column/row) was a full one. Measured to
      // move a whole-frame reading 1-2% on real crops
      // (dig_block_edge_norm.c).
      const double n_eff = fmax((double)buf[corner * comps + 4 * r + 2], 1.0);
      lambda[r] = g->ladder_lambda[r];
      energy[r] = (double)buf[corner * comps + 4 * r] / n_eff;
    }
    ok = TRUE;
  }
  dt_iop_gui_leave_critical_section(self);
  return ok;
}

// implementation-plan-8.md §5.4 last paragraph: the same p90/p99 read
// _box_block_percentiles gives a picked box, but over every block in the
// frame -- the baseline Phase 5's overlay draws the box's own reading
// against. A separate function rather than a parameter added to
// _spectrum_frame_wide above: that one is read on every graph redraw (it
// backs the always-on frame spectrum curve) and is O(1) per rung by
// construction (a single already-summed SAT corner); this one is a full
// per-rung sort over up to bw*bh blocks (<= 21600 per §5.4) -- called only
// from CT_PICK_PERCENTILE's own overlay case (Phase 5.2), not the hot path
// every redraw takes regardless of mode.
//
// Stops at the first rung where the whole frame overlaps fewer than
// CT_PERCENTILE_MIN_BLOCKS distinct super-blocks, for the same reason
// _ct_percentile_shape does: past that the ratio is 1 by construction, and
// drawing it would show the curve climbing to the top of the plot at the
// coarse end as if the frame's contrast were even there.
static gboolean _spectrum_frame_wide_percentiles(dt_iop_module_t *self,
                                                 double *const restrict p90,
                                                 double *const restrict p99,
                                                 int *const restrict nrungs)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;
  gboolean ok = FALSE;

  dt_iop_gui_enter_critical_section(self);
  const size_t sat_w = g->pd.width, sat_h = g->pd.height;
  const size_t comps = g->pd.components;
  const gboolean have_data =
    g->pd.buf && sat_w > 1 && sat_h > 1 && g->ladder_nrungs > 0
    && comps == (size_t)(4 * g->ladder_nrungs);

  if(have_data)
  {
    const size_t bw = sat_w - 1, bh = sat_h - 1;
    double *const restrict scratch = dt_alloc_align_double(MAX(bw * bh, (size_t)1));
    if(scratch)
    {
      const float *const restrict buf = g->pd.buf;
      *nrungs = 0;
      for(int r = 0; r < g->ladder_nrungs; r++)
      {
        const double step = exp2((double)(r / CT_SCALES_PER_OCTAVE));
        const size_t grp = _ladder_superblock_side(step);
        if(((bw + grp - 1) / grp) * ((bh + grp - 1) / grp) < CT_PERCENTILE_MIN_BLOCKS) break;

        size_t n = 0;
        for(size_t by = 0; by < bh; by++)
          for(size_t bx = 0; bx < bw; bx++)
            scratch[n++] = (double)buf[(by * sat_w + bx) * comps + 4 * r + 3];

        qsort(scratch, n, sizeof(double), _ct_cmp_double);
        p90[r] = _ct_percentile_sorted(scratch, n, 0.90);
        p99[r] = _ct_percentile_sorted(scratch, n, 0.99);
        *nrungs = r + 1;
      }
      dt_free_align(scratch);
      ok = TRUE;
    }
  }
  dt_iop_gui_leave_critical_section(self);
  return ok;
}

// draw one (lambda[], energy[]) polyline, in the current cairo source, over
// the graph's plotting area.
static void _draw_spectrum_curve(cairo_t *cr, const int width, const int height,
                                 const double *const restrict lambda,
                                 const double *const restrict energy,
                                 const int n, const double roi_long_edge, const double peak,
                                 const _ct_axis_t *const axis)
{
  if(n < 1) return;
  gboolean started = FALSE;
  for(int r = 0; r < n; r++)
  {
    const float x = _graph_lambda_to_x(lambda[r], roi_long_edge, axis) * width;
    const float y = height * (1.0f - _spectrum_energy_to_y(energy[r], peak));
    if(!started) { cairo_move_to(cr, x, y); started = TRUE; }
    else cairo_line_to(cr, x, y);
  }
  cairo_stroke(cr);
}

// implementation-plan-8.md §4.4/§5.4 Phase 5.2: percentile mode's own
// overlay draws q_r (p90/p99, a ratio in (0, 1]) directly, not through
// _spectrum_energy_to_y's log-energy mapping built for the always-on
// energy curves above -- linear top-to-bottom over [0, 1] is the whole
// range this quantity can ever take.
static void _draw_ratio_curve(cairo_t *cr, const int width, const int height,
                              const double *const restrict lambda,
                              const double *const restrict ratio,
                              const int n, const double roi_long_edge,
                              const _ct_axis_t *const axis)
{
  if(n < 1) return;
  gboolean started = FALSE;
  for(int r = 0; r < n; r++)
  {
    const float x = _graph_lambda_to_x(lambda[r], roi_long_edge, axis) * width;
    const float y = height * (1.0f - CLAMP((float)ratio[r], 0.0f, 1.0f));
    if(!started) { cairo_move_to(cr, x, y); started = TRUE; }
    else cairo_line_to(cr, x, y);
  }
  cairo_stroke(cr);
}

static void _draw_spectrum_overlay(cairo_t *cr, dt_iop_module_t *self,
                                   const int width, const int height,
                                   const _ct_axis_t *const axis)
{
  dt_iop_contrast_gui_data_t *const g = self->gui_data;

  double frame_lambda[CT_MAX_BANDS], frame_energy[CT_MAX_BANDS];
  int frame_nrungs = 0;
  const gboolean have_frame = _spectrum_frame_wide(self, frame_lambda, frame_energy, &frame_nrungs);

  dt_iop_gui_enter_critical_section(self);
  const gboolean have_pick = g->spectrum_valid;
  const int pick_nrungs = g->spectrum_nrungs;
  double pick_lambda[CT_MAX_BANDS], pick_energy[CT_MAX_BANDS];
  double fit_noise = 0.0, fit_self_similar = 0.0, fit_texture = 0.0, fit_tau = 0.0, fit_beta = 2.0;
  if(have_pick)
  {
    memcpy(pick_lambda, g->spectrum_lambda, sizeof(double) * pick_nrungs);
    memcpy(pick_energy, g->spectrum_energy, sizeof(double) * pick_nrungs);
    fit_noise = g->spectrum_noise;
    fit_self_similar = g->spectrum_self_similar;
    fit_texture = g->spectrum_texture;
    fit_tau = g->spectrum_tau;
    fit_beta = g->spectrum_beta;
  }
  // implementation-plan-8.md §4.4/§5.4 Phase 2.3: each adaptive mode's own
  // diagnostic, guarded the same way spectrum_* above is. Empty this phase
  // (nothing populates it before Phase 4/5) and for CT_PICK_FIXED forever --
  // the switch below is scaffolding for Phase 4/5 to fill in one case each.
  const int mode_overlay_n = g->mode_overlay_n;
  double mode_overlay[CT_MAX_BANDS];
  if(mode_overlay_n > 0) memcpy(mode_overlay, g->mode_overlay, sizeof(double) * mode_overlay_n);
  const double roi_long_edge = MAX(g->ladder_roi_in.width, g->ladder_roi_in.height);
  dt_iop_gui_leave_critical_section(self);

  if(!have_frame && !have_pick) return;

  const _ct_fit_t fit = { .noise = fit_noise, .self_similar = fit_self_similar,
                          .texture = fit_texture, .tau = fit_tau, .beta = fit_beta };

  double peak = 0.0;
  for(int r = 0; r < frame_nrungs; r++) peak = fmax(peak, frame_energy[r]);
  for(int r = 0; r < pick_nrungs; r++) peak = fmax(peak, pick_energy[r]);
  // implementation-plan-4.md §3.2: now that 3.1 evaluates the model in the
  // right variable, it can genuinely sit above both measured polylines --
  // sample it over the same range it is drawn across so peak reflects the
  // model too, rather than clipping it flat against the top of the plot.
  if(have_pick)
  {
    const double lo = pick_lambda[0], hi = pick_lambda[pick_nrungs - 1];
    for(int j = 0; j <= CT_GRAPH_RES; j++)
    {
      const double lambda = lo * exp2(log2(hi / fmax(lo, 1e-6)) * (double)j / (double)CT_GRAPH_RES);
      double S, N;
      _ct_fit_eval(&fit, lambda / CT_SIGMA_TO_LAMBDA / fmax(roi_long_edge, 1.0), &S, &N);
      peak = fmax(peak, S + N);
    }
  }
  if(peak <= 0.0) return;

  cairo_save(cr);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));

  if(have_frame)
  {
    cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                             darktable.bauhaus->graph_border.green,
                             darktable.bauhaus->graph_border.blue, 0.8);
    _draw_spectrum_curve(cr, width, height, frame_lambda, frame_energy, frame_nrungs,
                         roi_long_edge, peak, axis);
  }

  if(have_pick)
  {
    cairo_set_source_rgba(cr, darktable.bauhaus->color_fill.red,
                             darktable.bauhaus->color_fill.green,
                             darktable.bauhaus->color_fill.blue, 0.9);
    _draw_spectrum_curve(cr, width, height, pick_lambda, pick_energy, pick_nrungs,
                         roi_long_edge, peak, axis);

    // the fitted S(lambda) + N(lambda) model, sampled densely across the
    // picked box's own measured range, dashed to read as "model" rather
    // than "measurement" next to the polyline above.
    const double dashes[2] = { DT_PIXEL_APPLY_DPI(4.0), DT_PIXEL_APPLY_DPI(3.0) };
    cairo_set_dash(cr, dashes, 2, 0.0);
    gboolean started = FALSE;
    const double lo = pick_lambda[0], hi = pick_lambda[pick_nrungs - 1];
    for(int j = 0; j <= CT_GRAPH_RES; j++)
    {
      const double lambda = lo * exp2(log2(hi / fmax(lo, 1e-6)) * (double)j / (double)CT_GRAPH_RES);
      double S, N;
      // implementation-plan-4.md §3.1: fit->tau/self_similar/noise were solved
      // against _fit_curve_from_box's frame-relative sigma
      // (ladder_sigma[r]/long_edge); lambda here is in the ladder roi's own
      // pixels. Without the /roi_long_edge the model is evaluated L times too
      // far out: the self-similar term is scaled by L^(beta-2) -- +4.2 stops
      // at beta 2.4, +10.5 at beta 3.0 on a 10-stop axis, exact only at
      // beta = 2 -- and the noise and texture terms by 1/L^2, i.e. erased.
      // implementation-plan-2.md §3.2's comment predicted this conversion;
      // §4.2 made fit->tau frame-relative and never added it.
      _ct_fit_eval(&fit, lambda / CT_SIGMA_TO_LAMBDA / fmax(roi_long_edge, 1.0), &S, &N);
      const float x = _graph_lambda_to_x(lambda, roi_long_edge, axis) * width;
      const float y = height * (1.0f - _spectrum_energy_to_y(S + N, peak));
      if(!started) { cairo_move_to(cr, x, y); started = TRUE; }
      else cairo_line_to(cr, x, y);
    }
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
  }

  // implementation-plan-8.md §4.4/§5.4 Phase 2.3: each adaptive mode draws
  // its own observable on top of the measured/fitted curves above, once
  // Phase 4/5 populate mode_overlay[]/mode_overlay_n. CT_PICK_FIXED draws
  // nothing extra -- the default curve needs no additional diagnostic.
  if(mode_overlay_n > 0)
  {
    const _ct_picker_mode_t picker_mode = (_ct_picker_mode_t)dt_bauhaus_combobox_get(g->picker_mode);
    switch(picker_mode)
    {
      case CT_PICK_STRUCTURE:
      {
        // implementation-plan-8.md §4.4 Phase 4.2: the two rails first --
        // dashed, full width, graph_border like the other reference lines
        // on this graph -- so the kappa curve drawn on top of them reads
        // against a fixed scale rather than a floating one.
        const double dashes[2] = { DT_PIXEL_APPLY_DPI(2.0), DT_PIXEL_APPLY_DPI(2.0) };
        cairo_set_dash(cr, dashes, 2, 0.0);
        cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                                 darktable.bauhaus->graph_border.green,
                                 darktable.bauhaus->graph_border.blue, 0.6);
        const float y_gauss = height * (1.0f - _kappa_to_y(CT_KAPPA_GAUSSIAN));
        dt_draw_line(cr, 0, y_gauss, width, y_gauss);
        cairo_stroke(cr);
        const float y_struct = height * (1.0f - _kappa_to_y(CT_KAPPA_STRUCT));
        dt_draw_line(cr, 0, y_struct, width, y_struct);
        cairo_stroke(cr);
        cairo_set_dash(cr, NULL, 0, 0.0);

        // the box-wide kappa itself, the figure the curve is built from --
        // dotted, over `pick_lambda[]`'s same
        // rungs mode_overlay[] was measured on, in color_fill so it reads
        // as "this pick's own measurement" like the energy curve above,
        // distinguished from it by the dotted stroke and the different
        // (linear, fixed-range) y scale.
        const double dots[2] = { DT_PIXEL_APPLY_DPI(1.0), DT_PIXEL_APPLY_DPI(2.0) };
        cairo_set_dash(cr, dots, 2, 0.0);
        cairo_set_source_rgba(cr, darktable.bauhaus->color_fill.red,
                                 darktable.bauhaus->color_fill.green,
                                 darktable.bauhaus->color_fill.blue, 0.9);
        gboolean started = FALSE;
        const int n = MIN(mode_overlay_n, pick_nrungs);
        for(int r = 0; r < n; r++)
        {
          const float x = _graph_lambda_to_x(pick_lambda[r], roi_long_edge, axis) * width;
          const float y = height * (1.0f - _kappa_to_y(mode_overlay[r]));
          if(!started) { cairo_move_to(cr, x, y); started = TRUE; }
          else cairo_line_to(cr, x, y);
        }
        cairo_stroke(cr);
        cairo_set_dash(cr, NULL, 0, 0.0);
        break;
      }
      case CT_PICK_PERCENTILE:
      {
        // Phase 5.2: q_r per rung against the frame-wide q_r, between the
        // two rails the shape is read off -- all on the same linear [0,1]
        // ratio axis _draw_ratio_curve above draws. mode_overlay[]/
        // pick_lambda[] is this pick's own box q_r, published by
        // _mode_shape's CT_PICK_PERCENTILE case; the frame-wide curve is
        // cheap enough (O(blocks) per rung, no box clipping) to recompute
        // on every redraw rather than caching, the same way the always-on
        // frame spectrum curve above does.
        //
        // rails first, dashed and full width like structure mode's kappa
        // rails above, so the curves drawn on top read against a fixed
        // scale. Q_NOISE is the upper one on this axis (a ratio near 1 is
        // even, near 0 is concentrated), Q_STRUCT the lower.
        {
          const double dashes[2] = { DT_PIXEL_APPLY_DPI(2.0), DT_PIXEL_APPLY_DPI(2.0) };
          cairo_set_dash(cr, dashes, 2, 0.0);
          cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                                   darktable.bauhaus->graph_border.green,
                                   darktable.bauhaus->graph_border.blue, 0.6);
          const float y_noise = height * (1.0f - (float)CT_PERCENTILE_Q_NOISE);
          dt_draw_line(cr, 0, y_noise, width, y_noise);
          cairo_stroke(cr);
          const float y_struct = height * (1.0f - (float)CT_PERCENTILE_Q_STRUCT);
          dt_draw_line(cr, 0, y_struct, width, y_struct);
          cairo_stroke(cr);
          cairo_set_dash(cr, NULL, 0, 0.0);
        }

        double frame_p90[CT_MAX_BANDS], frame_p99[CT_MAX_BANDS];
        int frame_q_nrungs = 0;
        const gboolean have_frame_q =
          _spectrum_frame_wide_percentiles(self, frame_p90, frame_p99, &frame_q_nrungs);

        if(have_pick && mode_overlay_n > 0)
        {
          cairo_set_source_rgba(cr, darktable.bauhaus->color_fill.red,
                                   darktable.bauhaus->color_fill.green,
                                   darktable.bauhaus->color_fill.blue, 0.9);
          _draw_ratio_curve(cr, width, height, pick_lambda, mode_overlay, mode_overlay_n,
                            roi_long_edge, axis);
        }

        if(have_frame_q && frame_q_nrungs > 0)
        {
          double frame_q[CT_MAX_BANDS];
          for(int r = 0; r < frame_q_nrungs; r++)
            frame_q[r] = (frame_p99[r] > 0.0) ? frame_p90[r] / frame_p99[r] : 0.0;
          cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                                   darktable.bauhaus->graph_border.green,
                                   darktable.bauhaus->graph_border.blue, 0.8);
          _draw_ratio_curve(cr, width, height, frame_lambda, frame_q, frame_q_nrungs,
                            roi_long_edge, axis);
        }
        break;
      }
      case CT_PICK_FIXED:
      default:
        break;
    }
  }

  cairo_restore(cr);
}

static gboolean _area_draw(GtkWidget *widget, cairo_t *crf, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  const dt_iop_contrast_params_t *const p = self->params;
  const _ct_axis_t axis = _graph_axis(p);

  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  if(width <= 0 || height <= 0) return FALSE;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);

  cairo_surface_t *cst =
    dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, allocation.width, allocation.height);
  cairo_t *cr = cairo_create(cst);

  GtkStyleContext *context = gtk_widget_get_style_context(widget);
  gtk_render_background(context, cr, 0, 0, allocation.width, allocation.height);
  cairo_translate(cr, inset, inset);

  // 1. background grid: horizontal reference lines (visual density only,
  // unrelated to the x axis) plus one vertical line per octave boundary,
  // positioned on the same axis (§4.2, below) the curve and nodes use --
  // dt_draw_grid's own even spacing no longer matches now that the grid's
  // overhang takes up part of the width.
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(0.5));
  set_color(cr, darktable.bauhaus->graph_border);
  for(int k = 1; k < CT_BANDS; k++)
  {
    const float hy = k / (float)CT_BANDS * height;
    dt_draw_line(cr, 0, hy, width, hy);
    cairo_stroke(cr);
  }
  for(int k = 1; k < CT_BANDS; k++)
  {
    const float vx = _graph_raw_to_x((double)k / (double)CT_BANDS, &axis) * width;
    dt_draw_line(cr, vx, 0, vx, height);
    cairo_stroke(cr);
  }

  // 2. unresolvable-band shading -- bands beyond g->nbands (§1.5) don't
  // survive the current pipe scale and have no effect
  if(g->nbands < CT_BANDS)
  {
    const float x0 = _graph_raw_to_x((double)g->nbands / (double)CT_BANDS, &axis) * width;
    cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                             darktable.bauhaus->graph_border.green,
                             darktable.bauhaus->graph_border.blue, 0.4);
    cairo_rectangle(cr, x0, 0, width - x0, height);
    cairo_fill(cr);
  }

  // 2b. §6.3: shade the coarse end of the axis the last pick's own window
  // (§1.1) could not see -- coarse sits at x=0, same axis the unresolvable
  // shading above uses. The honest version of 6.1/6.2's refusal/warning:
  // this is *why* the pick answered what it did. The coarsest rung that
  // survived the window (published as the last entry of g->spectrum_lambda,
  // already the windowed set per §1.3) stands in for the window's own
  // cutoff -- close enough, since nothing coarser than it was ever queried.
  {
    dt_iop_gui_enter_critical_section(self);
    const gboolean have_pick_window = g->spectrum_valid && g->spectrum_nrungs > 0;
    const double pick_coarsest_lambda =
      have_pick_window ? g->spectrum_lambda[g->spectrum_nrungs - 1] : 0.0;
    const double window_roi_long_edge = MAX(g->ladder_roi_in.width, g->ladder_roi_in.height);
    dt_iop_gui_leave_critical_section(self);

    if(have_pick_window)
    {
      const float x1 = _graph_lambda_to_x(pick_coarsest_lambda, window_roi_long_edge, &axis) * width;
      if(x1 > 0.0f)
      {
        cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                                 darktable.bauhaus->graph_border.green,
                                 darktable.bauhaus->graph_border.blue, 0.4);
        cairo_rectangle(cr, 0, 0, x1, height);
        cairo_fill(cr);
      }
    }
  }

  // 2c. implementation-plan-3.md §4.3: shade continuously by how much of a
  // wavelength's energy the projection grid's nine bands can even address
  // (1 - sum_k H_k, §3.2's own row weight) -- full shading where the basis
  // has nothing there, none where sum_k H_k is 1. Distinct from both
  // shadings above: item 2 says "too fine for this pipe scale", item 2b
  // says "outside this pick's own window", this one says "outside what the
  // nine-band ladder itself can ever reach", continuous rather than a
  // stepped rectangle since the quantity itself is continuous. Sampled at
  // CT_GRAPH_RES columns across the axis's own sigma span, same style as
  // the curve's own sampling a few lines down.
  {
    float sigma_ladder[CT_BANDS];
    _ct_band_sigma(sigma_ladder, p->scale_shift);
    double grid_lo, grid_hi;
    _ct_grid_bounds(sigma_ladder, &grid_lo, &grid_hi);
    for(int i = 0; i < CT_GRAPH_RES; i++)
    {
      // t=0 is screen-left/coarse (grid_hi, the larger sigma), t=1 is
      // screen-right/fine (grid_lo) -- the *opposite* sweep direction from
      // color_picker_apply's own sigma_grid[], which runs fine-to-coarse as
      // its index increases (that array has no notion of screen position).
      const double t = ((double)i + 0.5) / (double)CT_GRAPH_RES;
      const double s = grid_hi * exp2(log2(grid_lo / grid_hi) * t);
      const double lambda = s * CT_SIGMA_TO_LAMBDA;
      const double coverage = CLAMP(_ct_band_coverage(lambda, sigma_ladder, CT_BANDS), 0.0, 1.0);
      const double alpha = 0.4 * (1.0 - coverage);
      if(alpha <= 0.002) continue;
      cairo_set_source_rgba(cr, darktable.bauhaus->graph_border.red,
                               darktable.bauhaus->graph_border.green,
                               darktable.bauhaus->graph_border.blue, alpha);
      const float xa = (float)i / CT_GRAPH_RES * width;
      const float xb = (float)(i + 1) / CT_GRAPH_RES * width;
      cairo_rectangle(cr, xa, 0, xb - xa, height);
      cairo_fill(cr);
    }
  }

  // 3. baseline at gain 1.0
  const float baseline_y = height * (1.0f - _graph_gain_to_yfrac(1.0f));
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
  dt_draw_line(cr, 0, baseline_y, width, baseline_y);
  cairo_stroke(cr);

  // 3b. implementation-plan-3.md §4.1: the EQUALIZE envelope rails, dashed,
  // in the same weight as the baseline above -- a node railed against one of
  // these is now a node visibly touching a drawn line.
  //
  // §4.5 (was Phase 6.2): decided this is the whole answer to "what does a
  // node on the envelope look like", nothing further to draw, once these
  // rails and the log axis (§4.1) are both in place -- re-confirmed after
  // §4.3/§4.4 landed too, since a node can now also be dashed (extrapolated,
  // §4.4) at the same time it touches a rail. The two marks read as separate
  // questions ("was this measured" vs. "is this at its limit") on different
  // parts of the node (outline dash pattern vs. y position against a drawn
  // line), so no combined case needs handling.
  {
    const float lo_y = height * (1.0f - _graph_gain_to_yfrac(CT_EQUALIZE_GAIN_LO));
    const float hi_y = height * (1.0f - _graph_gain_to_yfrac(CT_EQUALIZE_GAIN_HI));
    const double dashes[2] = { DT_PIXEL_APPLY_DPI(4.0), DT_PIXEL_APPLY_DPI(3.0) };
    set_color(cr, darktable.bauhaus->graph_fg);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
    cairo_set_dash(cr, dashes, 2, 0.0);
    dt_draw_line(cr, 0, lo_y, width, lo_y);
    dt_draw_line(cr, 0, hi_y, width, hi_y);
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
  }

  // 4. the measured spectrum (§3.2): frame-wide ladder always in the
  // background, the last pick's own spectrum + fitted model on top of it.
  _draw_spectrum_overlay(cr, self, width, height, &axis);

  // 5. the curve: monotone cubic through the nine nodes -- this is the
  // *shape*, i.e. exactly what dragging a node edits (p->band[k]), not what
  // reaches the pixels once the master gain is applied (see 5b below).
  _graph_curve_from_params(g->curve, p, &axis);
  float xs[CT_GRAPH_RES], ys[CT_GRAPH_RES];
  dt_draw_curve_calc_values(g->curve, 0.0f, 1.0f, CT_GRAPH_RES, xs, ys);
  set_color(cr, darktable.bauhaus->graph_fg);
  cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(2.0));
  cairo_move_to(cr, 0, height * (1.0f - ys[0]));
  for(int i = 1; i < CT_GRAPH_RES; i++)
    cairo_line_to(cr, i * width / (float)(CT_GRAPH_RES - 1), height * (1.0f - ys[i]));
  cairo_stroke(cr);

  // 5b. implementation-plan-6.md §6 Phase 4.1: the *effective* gain overlay,
  // 1 + master*(shape-1) -- what §1's bug report actually judged. Dashed,
  // since it's derived from the shape curve rather than directly editable
  // (this file's existing convention: solid = editable/measured, dashed =
  // derived -- e.g. the spectrum overlay's fitted model below). Skipped at
  // master == 1 exactly, where it would trace the shape curve on top of
  // itself and add nothing to look at. g->curve is scratch state private to
  // this draw call (nothing after this point reads it), so reusing it here
  // rather than allocating a second curve is safe.
  if(p->gain_local_contrast != 1.0f)
  {
    for(int k = 0; k < CT_BANDS; k++)
      dt_draw_curve_set_point(g->curve, k, _graph_node_x(k, &axis),
                              _graph_gain_to_yfrac(_graph_effective_gain(k, p->band[k], p->gain_local_contrast,
                                                                        p->scale_shift)));
    float exs[CT_GRAPH_RES], eys[CT_GRAPH_RES];
    dt_draw_curve_calc_values(g->curve, 0.0f, 1.0f, CT_GRAPH_RES, exs, eys);
    const double eff_dashes[2] = { DT_PIXEL_APPLY_DPI(3.0), DT_PIXEL_APPLY_DPI(2.0) };
    cairo_set_dash(cr, eff_dashes, 2, 0.0);
    set_color(cr, darktable.bauhaus->graph_fg);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
    cairo_move_to(cr, 0, height * (1.0f - eys[0]));
    for(int i = 1; i < CT_GRAPH_RES; i++)
      cairo_line_to(cr, i * width / (float)(CT_GRAPH_RES - 1), height * (1.0f - eys[i]));
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
  }

  // 5c. implementation-plan-7.md §4.2: the ceiling itself, third curve --
  // _ct_band_ceiling per band, purely frame-relative (needs only k and
  // scale_shift, §4.4) -- like 5b just above, derived directly from
  // self->params, not published from the pipe (see _ct_band_ceiling's own
  // comment). Drawn unconditionally, like the envelope rails (3b): with the
  // per-band knee (§4.1(d)) this is where 5b's dashed line is headed once a
  // band's own R_k starts binding, and that is worth seeing before the
  // slider is ever touched, not only after. Dotted rather than dashed so it
  // reads as a third, distinct line rather than a second copy of 5b's dash
  // style; graph_border rather than graph_fg keeps it visually behind the
  // two editable/derived curves.
  {
    for(int k = 0; k < CT_BANDS; k++)
      dt_draw_curve_set_point(g->curve, k, _graph_node_x(k, &axis),
                              _graph_gain_to_yfrac((float)_ct_band_ceiling(k, p->scale_shift)));
    float cxs[CT_GRAPH_RES], cys[CT_GRAPH_RES];
    dt_draw_curve_calc_values(g->curve, 0.0f, 1.0f, CT_GRAPH_RES, cxs, cys);
    const double ceiling_dots[2] = { DT_PIXEL_APPLY_DPI(1.0), DT_PIXEL_APPLY_DPI(2.0) };
    cairo_set_dash(cr, ceiling_dots, 2, 0.0);
    set_color(cr, darktable.bauhaus->graph_border);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
    cairo_move_to(cr, 0, height * (1.0f - cys[0]));
    for(int i = 1; i < CT_GRAPH_RES; i++)
      cairo_line_to(cr, i * width / (float)(CT_GRAPH_RES - 1), height * (1.0f - cys[i]));
    cairo_stroke(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
  }

  // 6. node bars + bullets
  //
  // implementation-plan-3.md §4.4: which nodes the last pick's own window
  // (§1.1) actually measured, vs. which ones the fit's power law only
  // extrapolates to -- the same window §2b already shades on the axis
  // itself, now also marked on the nodes it covers. Raw x, not screen x:
  // this is a comparison of positions along the ladder, so it needs to
  // happen before the axis's own screen clamping.
  dt_iop_gui_enter_critical_section(self);
  const gboolean have_pick_window = g->spectrum_valid && g->spectrum_nrungs > 0;
  double window_lo_raw = 0.0, window_hi_raw = 0.0;  // coarsest .. finest measured, raw x
  if(have_pick_window)
  {
    const double long_edge = MAX(g->ladder_roi_in.width, g->ladder_roi_in.height);
    window_lo_raw = _spectrum_lambda_to_raw_x(g->spectrum_lambda[g->spectrum_nrungs - 1], long_edge);
    window_hi_raw = _spectrum_lambda_to_raw_x(g->spectrum_lambda[0], long_edge);
  }
  dt_iop_gui_leave_critical_section(self);

  for(int k = 0; k < CT_BANDS; k++)
  {
    const float xn = _graph_node_x(k, &axis) * width;
    const float yfrac = _graph_gain_to_yfrac(p->band[k]);
    const float yn = height * (1.0f - yfrac);
    const double node_raw = ((double)k + 0.5) / (double)CT_BANDS;
    const gboolean extrapolated =
      have_pick_window && (node_raw < window_lo_raw || node_raw > window_hi_raw);

    // implementation-plan-6.md §6 Phase 4.1: a node sitting innocently inside
    // the envelope (e.g. the EQUALIZE floor at 0.30) can still have master
    // push its *effective* gain at or below zero, inverting that octave's
    // detail (§2.2) -- the node bullet itself is what a user actually looks
    // at when judging a curve, so this is marked here, not only on the 5b
    // overlay curve, and it overrides the extrapolated color (not the dash,
    // which is a separate question per §4.5).
    const float effective_gain = _graph_effective_gain(k, p->band[k], p->gain_local_contrast,
                                                       p->scale_shift);
    const gboolean negative_effective = effective_gain <= 0.0f;

    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(6));
    set_color(cr, darktable.bauhaus->color_fill);
    dt_draw_line(cr, xn, baseline_y, xn, yn);
    cairo_stroke(cr);

    const gboolean active = (k == g->hover_band || k == g->drag_band);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
    cairo_arc(cr, xn, yn, DT_PIXEL_APPLY_DPI(active ? 5.0 : 3.5), 0.0, 2.0 * M_PI);
    if(extrapolated)
    {
      const double dashes[2] = { DT_PIXEL_APPLY_DPI(1.5), DT_PIXEL_APPLY_DPI(1.5) };
      cairo_set_dash(cr, dashes, 2, 0.0);
    }
    if(negative_effective)
      cairo_set_source_rgba(cr, 0.8, 0.1, 0.1, 1.0);
    else if(extrapolated)
      set_color(cr, darktable.bauhaus->graph_border);
    else
      set_color(cr, darktable.bauhaus->graph_fg);
    cairo_stroke_preserve(cr);
    cairo_set_dash(cr, NULL, 0, 0.0);
    if(k == g->drag_band)
      set_color(cr, darktable.bauhaus->graph_fg);
    else
      set_color(cr, darktable.bauhaus->graph_bg);
    cairo_fill(cr);

    // §1.6: this band's own texture is the mask currently shown -- the link
    // that answers "you can only visualize the main detail scale"
    if(k == g->details_display)
    {
      cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.5));
      set_color(cr, darktable.bauhaus->graph_fg);
      cairo_arc(cr, xn, yn, DT_PIXEL_APPLY_DPI(7.5), 0.0, 2.0 * M_PI);
      cairo_stroke(cr);
    }

    // implementation-plan-6.md §6 Phase 4.1's acceptance bullet is explicit:
    // a railed node must read as *negative*, not as a node sitting innocently
    // on its rail -- the log axis can't place a negative y at all (§4.1
    // above already collapses it to the floor), so the actual signed number
    // is printed at the floor instead, in the same warning color as the node.
    if(negative_effective)
    {
      char eff_buf[16];
      snprintf(eff_buf, sizeof(eff_buf), "%.2f", (double)effective_gain);
      PangoFontDescription *eff_desc = dt_gui_get_font();
      pango_font_description_set_absolute_size(eff_desc, 0.08 * height * PANGO_SCALE);
      PangoLayout *eff_layout = pango_cairo_create_layout(cr);
      pango_layout_set_font_description(eff_layout, eff_desc);
      cairo_set_source_rgba(cr, 0.8, 0.1, 0.1, 1.0);
      pango_layout_set_text(eff_layout, eff_buf, -1);
      PangoRectangle eff_ink;
      pango_layout_get_pixel_extents(eff_layout, &eff_ink, NULL);
      cairo_move_to(cr, xn - eff_ink.width / 2.0, height - eff_ink.height - DT_PIXEL_APPLY_DPI(2));
      pango_cairo_show_layout(cr, eff_layout);
      g_object_unref(eff_layout);
      pango_font_description_free(eff_desc);
    }
  }

  // §1.6: print the divisor a texture mask view was normalized by, so the
  // view stays quantitative rather than just "brighter means more"
  if(g->details_display == DT_CT_MASK_DETAIL
     || (g->details_display >= 0 && g->details_display < CT_BANDS))
  {
    dt_iop_gui_enter_critical_section(self);
    const float divisor = g->mask_divisor;
    dt_iop_gui_leave_critical_section(self);

    char buf[32];
    snprintf(buf, sizeof(buf), "×%.4g", (double)divisor);
    PangoFontDescription *div_desc = dt_gui_get_font();
    pango_font_description_set_absolute_size(div_desc, 0.09 * height * PANGO_SCALE);
    PangoLayout *div_layout = pango_cairo_create_layout(cr);
    pango_layout_set_font_description(div_layout, div_desc);
    set_color(cr, darktable.bauhaus->graph_fg);
    pango_layout_set_text(div_layout, buf, -1);
    PangoRectangle div_ink;
    pango_layout_get_pixel_extents(div_layout, &div_ink, NULL);
    cairo_move_to(cr, width - div_ink.width - DT_PIXEL_APPLY_DPI(2),
                 height - div_ink.height - DT_PIXEL_APPLY_DPI(2));
    pango_cairo_show_layout(cr, div_layout);
    g_object_unref(div_layout);
    pango_font_description_free(div_desc);
  }

  // axis labels
  PangoFontDescription *desc = dt_gui_get_font();
  pango_font_description_set_absolute_size(desc, 0.09 * height * PANGO_SCALE);
  PangoLayout *layout = pango_cairo_create_layout(cr);
  pango_layout_set_font_description(layout, desc);
  set_color(cr, darktable.bauhaus->graph_fg);

  pango_layout_set_text(layout, _("coarse"), -1);
  cairo_move_to(cr, DT_PIXEL_APPLY_DPI(2), DT_PIXEL_APPLY_DPI(2));
  pango_cairo_show_layout(cr, layout);

  PangoRectangle ink;
  pango_layout_set_text(layout, _("fine"), -1);
  pango_layout_get_pixel_extents(layout, &ink, NULL);
  cairo_move_to(cr, width - ink.width - DT_PIXEL_APPLY_DPI(2), DT_PIXEL_APPLY_DPI(2));
  pango_cairo_show_layout(cr, layout);

  g_object_unref(layout);
  pango_font_description_free(desc);

  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return FALSE;
}

static void _area_motion(GtkEventControllerMotion *controller,
                         gdouble x, gdouble y,
                         dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  GtkWidget *widget = dt_gui_get_widget(controller);
  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  const double gx = x - inset, gy = y - inset;
  const _ct_axis_t axis = _graph_axis((const dt_iop_contrast_params_t *)self->params);

  g->hover_band = (gx >= 0 && gx <= width) ? _graph_band_at(width, gx, &axis) : -1;

  if(g->dragging && g->drag_band >= 0)
    _area_set_band(g, g->drag_band, _graph_gain_at(height, gy));

  gtk_widget_queue_draw(widget);
}

static void _area_leave(GtkEventControllerMotion *controller, dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  g->hover_band = -1;
  gtk_widget_queue_draw(dt_gui_get_widget(controller));
}

static void _area_button_press(GtkGestureSingle *gesture,
                               gint n_press,
                               gdouble x, gdouble y,
                               dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  GtkWidget *widget = dt_gui_get_widget(gesture);
  const guint button = gtk_gesture_single_get_current_button(gesture);

  if(button == GDK_BUTTON_MIDDLE)
  {
    const gchar *current = gtk_stack_get_visible_child_name(g->stack);
    gtk_stack_set_visible_child_name(g->stack,
                                     g_strcmp0(current, "graph") == 0 ? "sliders" : "graph");
    return;
  }

  if(button != GDK_BUTTON_PRIMARY) return;

  int inset, width, height;
  _graph_geometry(widget, &inset, &width, &height);
  const _ct_axis_t axis = _graph_axis((const dt_iop_contrast_params_t *)self->params);
  const int k = _graph_band_at(width, x - inset, &axis);

  // ctrl+click selects that band's mask view (§1.6) rather than dragging its
  // value -- the two would otherwise fight over the same click.
  if(dt_modifier_is(dt_key_modifier_state(), GDK_CONTROL_MASK))
  {
    if(!_mask_display_blocked(self))
      _apply_details_display(self, (dt_iop_details_display_t)k);
    gtk_widget_queue_draw(widget);
    return;
  }

  if(n_press >= 2)
  {
    const dt_iop_contrast_params_t *const def = self->default_params;
    _area_set_band(g, k, def->band[k]);
    gtk_widget_queue_draw(widget);
    return;
  }

  g->dragging = TRUE;
  g->drag_band = k;
  _area_set_band(g, k, _graph_gain_at(height, y - inset));
  gtk_widget_queue_draw(widget);
}

static void _area_button_release(GtkGestureSingle *gesture,
                                 gint n_press,
                                 gdouble x, gdouble y,
                                 dt_iop_module_t *self)
{
  if(gtk_gesture_single_get_current_button(gesture) != GDK_BUTTON_PRIMARY) return;
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  g->dragging = FALSE;
  g->drag_band = -1;
  gtk_widget_queue_draw(dt_gui_get_widget(gesture));
}

static void _area_scrolled(GtkEventControllerScroll *controller,
                           gdouble dx, gdouble dy,
                           dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  if(g->hover_band < 0 || dy == 0.0) return;

  const dt_iop_contrast_params_t *const p = self->params;
  const float step = dt_modifier_eq(controller, GDK_CONTROL_MASK) ? 0.01f : 0.05f;
  _area_set_band(g, g->hover_band, p->band[g->hover_band] - (float)dy * step);
  gtk_widget_queue_draw(dt_gui_get_widget(controller));
}

enum
{
  DT_ACTION_EFFECT_CT_RESET = DT_ACTION_EFFECT_RESET,
};

static const dt_action_element_def_t _action_elements_ct[]
  = { { N_("band 1 (coarsest)"), dt_action_effect_value },
      { N_("band 2"), dt_action_effect_value },
      { N_("band 3"), dt_action_effect_value },
      { N_("band 4"), dt_action_effect_value },
      { N_("band 5"), dt_action_effect_value },
      { N_("band 6"), dt_action_effect_value },
      { N_("band 7"), dt_action_effect_value },
      { N_("band 8"), dt_action_effect_value },
      { N_("band 9 (finest)"), dt_action_effect_value },
      { } };

static float _action_process_ct(gpointer target,
                                const dt_action_element_t element,
                                const dt_action_effect_t effect,
                                float move_size)
{
  if(element < 0 || element >= CT_BANDS) return DT_ACTION_NOT_VALID;

  dt_iop_module_t *self = g_object_get_data(G_OBJECT(target), "iop-instance");
  dt_iop_contrast_gui_data_t *g = self->gui_data;
  dt_iop_contrast_params_t *p = self->params;
  const dt_iop_contrast_params_t *const d = self->default_params;

  if(DT_PERFORM_ACTION(move_size))
  {
    switch(effect)
    {
      case DT_ACTION_EFFECT_CT_RESET:
        _area_set_band(g, element, d->band[element]);
        break;
      case DT_ACTION_EFFECT_DOWN:
        move_size *= -1;
      case DT_ACTION_EFFECT_UP:
        _area_set_band(g, element, p->band[element] + move_size / 100.0f);
        break;
      default:
        break;
    }
    gtk_widget_queue_draw(GTK_WIDGET(g->area));
  }

  return p->band[element] + DT_VALUE_PATTERN_PLUS_MINUS;
}

static const dt_action_def_t _action_def_ct
  = { N_("detail ladder"),
      _action_process_ct,
      _action_elements_ct,
      NULL };

void gui_init(dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = IOP_GUI_ALLOC(contrast);
  g->details_display = DT_CT_MASK_OFF;
  g->mask_divisor = 1.0f;
  dt_preview_data_alloc(&g->pd, self);  // §2.2: the frame-wide ladder's SAT tables
  dt_preview_data_alloc(&g->band_pd, self);  // §3.1: the module's own per-band tables
  g->band_pd.components = 2 * CT_BANDS + 1;  // fixed forever, unlike the ladder's nrungs -- no resize dance; +1 is §8.2's shared block-count column

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_UI_PIPE_FINISHED, _ui_pipe_done);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished_retry_pick);

  // Main container
  self->widget = dt_gui_vbox();

  // Local boost slider
  // plan-7 §6 Phase 3.2: displayed as a direct multiplier ("1.00x" at
  // neutral) rather than a percent offset from neutral -- the underlying
  // quantity already is one (effective gain = 1 + this * (band gain - 1)),
  // and "+1000%" at the new, much wider hard range reads worse than "10x"
  // for the same number.
  g->gain_local_contrast = dt_bauhaus_slider_from_params(self, "gain_local_contrast");
  dt_bauhaus_slider_set_soft_range(g->gain_local_contrast, 0.0, 10.0);
  dt_bauhaus_slider_set_digits(g->gain_local_contrast, 2);
  dt_bauhaus_slider_set_format(g->gain_local_contrast, "x");
  gtk_widget_set_tooltip_text(g->gain_local_contrast,
                              _("scales the picked (or hand-drawn) curve up or down: each\n"
                                "band's effective gain is 1 + this * (band gain - 1).\n"
                                "past a perceptual countershading limit, which varies by band and\n"
                                "is drawn as the graph's third, dotted curve, a band's own effective\n"
                                "gain bends off smoothly rather than climbing further -- see the\n"
                                "graph for which bands still have headroom."));
  dt_bauhaus_widget_set_quad(g->gain_local_contrast, self, dtgtk_cairo_paint_showmask, TRUE, show_details_callback,
                             _("visualize the accumulated correction -- what this module is doing.\n"
                               "ctrl+click: visualize the raw, un-gained detail sum (v1's behaviour)."));

  // Filter settings section
  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "filter settings")));

  // the graph (§1.4): nine nodes, one per octave, drawn as a curve and
  // draggable directly; a GtkStack toggled by middle-click on the graph
  // swaps it for the plain slider list, which is what the shortcut system
  // and anyone chasing an exact value still reach for.
  g->nbands = CT_BANDS;  // until the first preview pass publishes the real count (§1.5)
  g->hover_band = -1;
  g->drag_band = -1;
  g->dragging = FALSE;
  g->curve = dt_draw_curve_new(0.0, 1.0, MONOTONE_HERMITE);
  {
    const dt_iop_contrast_params_t *const def = (dt_iop_contrast_params_t *)self->default_params;
    const _ct_axis_t axis = _graph_axis(def);
    for(int k = 0; k < CT_BANDS; k++)
      dt_draw_curve_add_point(g->curve, _graph_node_x(k, &axis), _graph_gain_to_yfrac(def->band[k]));
  }

  g->area = GTK_DRAWING_AREA(dt_ui_resize_wrap
                             (NULL, 0, "plugins/darkroom/contrastadv/graphheight"));
  g_object_set_data(G_OBJECT(g->area), "iop-instance", self);
  dt_action_define_iop(self, NULL, N_("graph"), GTK_WIDGET(g->area), &_action_def_ct);
  g_signal_connect(G_OBJECT(g->area), "draw", G_CALLBACK(_area_draw), self);
  dt_gui_connect_click(g->area, _area_button_press, _area_button_release, self);
  dt_gui_connect_motion(g->area, _area_motion, _area_motion, _area_leave, self);
  dt_gui_connect_scroll(g->area, GTK_EVENT_CONTROLLER_SCROLL_BOTH_AXES
                               | GTK_EVENT_CONTROLLER_SCROLL_DISCRETE, _area_scrolled, self);
  _area_set_tooltip(g);  // §6.3: initial state, until the first _ui_pipe_done updates it

  // one slider per band, labeled by the node's nominal size -- computed
  // rather than nine near-identical translated strings, per
  // implementation-plan.md §1.1.
  GtkWidget *sliders_box = dt_gui_vbox();
  dt_iop_module_t *section = DT_IOP_SECTION_FOR_PARAMS(self, NC_("section", "bands"), sliders_box);
  for(int k = 0; k < CT_BANDS; k++)
  {
    char param[16];
    snprintf(param, sizeof(param), "band[%d]", k);
    g->band[k] = dt_bauhaus_slider_from_params(section, param);
    dt_bauhaus_slider_set_soft_range(g->band[k], 0.0, 2.0);
    dt_bauhaus_slider_set_digits(g->band[k], 2);
    dt_bauhaus_slider_set_format(g->band[k], "%");
    dt_bauhaus_slider_set_factor(g->band[k], 100.0);
    dt_bauhaus_slider_set_offset(g->band[k], -100.0);

    char label[64];
    snprintf(label, sizeof(label), _("detail size ~ %.3g%%"), 100.0 * exp2(-(CT_BAND_D0 + k)));
    dt_bauhaus_widget_set_label(g->band[k], NULL, label);

    // §1.6: this quad shows this band's own raw detail texture, whatever its
    // gain -- "ct-band" is what show_details_callback reads to know which.
    g_object_set_data(G_OBJECT(g->band[k]), "ct-band", GINT_TO_POINTER(k));
    dt_bauhaus_widget_set_quad(g->band[k], self, dtgtk_cairo_paint_showmask, TRUE, show_details_callback,
                               _("visualize this band's own detail texture\n"
                                 "(the same as ctrl+clicking its node in the graph)"));
  }

  g->stack = GTK_STACK(gtk_stack_new());
  gtk_stack_set_homogeneous(g->stack, FALSE);
  gtk_stack_add_named(g->stack, GTK_WIDGET(g->area), "graph");
  gtk_stack_add_named(g->stack, sliders_box, "sliders");
  gtk_stack_set_visible_child_name(g->stack, "graph");
  dt_action_define_iop(self, NULL, N_("sliders"), GTK_WIDGET(g->stack), NULL);
  dt_gui_box_add(self->widget, g->stack);

  // implementation-plan-8.md §4.1/§4.3: a GUI preference, not a param --
  // placed directly above the picker row below so the mode and the button
  // that uses it are adjacent. Selecting an entry has no effect on its own;
  // it changes what the *next pick* does.
  g->picker_mode = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->picker_mode, NULL, N_("picker mode"));
  dt_bauhaus_combobox_add(g->picker_mode, _("fixed curve"));
  dt_bauhaus_combobox_add(g->picker_mode, _("structured detail"));
  dt_bauhaus_combobox_add(g->picker_mode, _("local contrast levels"));
  dt_bauhaus_combobox_set(g->picker_mode,
                          CLAMP(dt_conf_get_int(CT_PICKER_MODE_CONF), CT_PICK_FIXED, CT_PICK_PERCENTILE));
  dt_gui_box_add(self->widget, g->picker_mode);
  gtk_widget_set_tooltip_text
    (g->picker_mode,
     _("what the area picker measures, and how it turns that into a curve:\n"
       "fixed curve -- ignores the picked area's own texture and writes the\n"
       "  same default hump every time. the one thing an old-style pick did\n"
       "  that you might still want on its own.\n"
       "structured detail -- boosts the bands whose energy in the picked area\n"
       "  is concentrated in edges and lines rather than spread out like\n"
       "  texture or noise. declines only on a pick that reads as noise\n"
       "  at every size.\n"
       "local contrast levels -- boosts the bands whose local contrast is\n"
       "  spatially uneven across the picked area, usually leaning toward\n"
       "  the finer end on real content. declines where local contrast is\n"
       "  as even as noise at every size."));
  g_signal_connect(G_OBJECT(g->picker_mode), "value-changed", G_CALLBACK(_picker_mode_callback), self);

  g->scale_shift = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                       dt_bauhaus_slider_from_params(self, "scale_shift"));
  gtk_widget_set_tooltip_text(g->scale_shift,
     _("shifts every band's node together, finer or coarser.\n"
       "half a step moves the whole ladder by half an octave."));
  // implementation-plan-4.md §6.1: this quad no longer moves scale_shift --
  // it never has, since plan-3 Phase 6 decided against ladder placement and
  // routed the picker through band[]/gain_local_contrast instead (§6.2's own
  // comment below). The old wording promised a "shift the ladder" effect
  // this control does not have, and warned about an outcome
  // (scale_shift-less picks "coming back at the finest setting") that cannot
  // occur -- what a noisy pick actually produces is the dt_control_log
  // warning quoted below.
  // implementation-plan-8.md §4.3: rewritten -- it used to promise "set each
  // band's gain from what was found", which is exactly what the fixed mode
  // does not do (plan-6 §5B.2 onward); what a pick does now is set by the
  // picker-mode dropbox above.
  dt_bauhaus_widget_set_quad_tooltip
    (g->scale_shift,
     _("pick an area: measure it and shape the curve according to the picker\n"
       "mode set above.\n"
       "click to use the whole frame, then drag on the image to work from the\n"
       "subject that matters instead. a small box cannot report structure\n"
       "larger than itself, so pick over as much of the texture as you want\n"
       "counted.\n"
       "an area with nothing in it to enhance -- clear sky, an out-of-focus\n"
       "background -- is declined rather than guessed at. if a pick over deep\n"
       "shadow logs \"looks like noise\", raise the noise bias below and pick\n"
       "again."));

  g->edge_protection = dt_bauhaus_slider_from_params(self, "edge_protection");
  dt_bauhaus_slider_set_soft_range(g->edge_protection, -2.0, 2.0);
  dt_bauhaus_slider_set_digits(g->edge_protection, 2);
  dt_bauhaus_slider_set_format(g->edge_protection, "%");
  dt_bauhaus_slider_set_factor(g->edge_protection, 100.0);
  gtk_widget_set_tooltip_text(g->edge_protection, _("adjust the edge sensitivity of the filter\n"
                                                    "higher = more edge preservation\n"
                                                    "lower = smoother transitions, but may lead to halos around edges"));

  g->filter_iterations = dt_bauhaus_slider_from_params(self, "filter_iterations");
  dt_bauhaus_slider_set_soft_range(g->filter_iterations, 1, 5);
  gtk_widget_set_tooltip_text(g->filter_iterations, _("number of passes of the guided filter to apply\n"
       "helps diffusing the edges of the filter at the expense of speed"));

  g->noise_bias = dt_bauhaus_slider_from_params(self, "noise_bias");
  dt_bauhaus_slider_set_soft_range(g->noise_bias, 0.0, 0.2);
  dt_bauhaus_slider_set_digits(g->noise_bias, 4);
  dt_bauhaus_slider_set_step(g->noise_bias, 0.0001);
  gtk_widget_set_tooltip_text(g->noise_bias, _("add bias to reduce shadow noise amplification.\n"
                                               "only affects dark parts of the image."));
}

// §1.8: dt_iop_gui_cleanup_module() (develop/imageop.c) already
// DT_CONTROL_SIGNAL_DISCONNECT_ALL()s every DT_CONTROL_SIGNAL_HANDLE
// registered in gui_init before calling this, so the ui-pipe-done handler
// needs no disconnect of its own here -- verified against the framework
// rather than assumed, and matches every other iop's gui_cleanup in this
// tree (atrous, toneequal).
void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_contrast_gui_data_t *g = self->gui_data;

  // drop any pick this module still owns, mirroring colorequal.c/toneequal.c
  // -- if the module is torn down while a pick is armed or in flight,
  // nothing else clears the global color picker for us.
  self->request_color_pick = DT_REQUEST_COLORPICK_OFF;

  dt_draw_curve_destroy(g->curve);
  dt_preview_data_free(&g->pd);  // §2.2
  dt_preview_data_free(&g->band_pd);  // §3.1
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
