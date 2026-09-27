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
 * cmocka unit tests for the math of iop/lightsurface.c (epic M0-M5),
 * against golden data from the Python reference implementation
 * (lightsurface_golden.h), plus the exact identity at default params.
 *
 * Please see README.md for more detailed documentation.
 */
#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include <cmocka.h>

// lightsurface.c is compiled into this file to reach its static functions
#include "iop/lightsurface.c"

#include "lightsurface_golden.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

// absolute tolerance for values in EV; float32 carries about 7 digits, so
// this allows a few rounding steps of difference from the numpy reference
#define E 1e-5f

// a fixed-seed generator, so a failure is reproducible
static uint32_t _rng_state = 20260927u;

static float _rand_uniform(const float lo, const float hi)
{
  _rng_state = _rng_state * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((_rng_state >> 8) * (1.0f / 16777216.0f));
}

static void _data_init(dt_iop_lightsurface_data_t *const d)
{
  memset(d, 0, sizeof(*d));
  // non-zero by design; no math reads it
  d->model_hash = 0x12345678u;
}

static int group_setup(void **state)
{
  // dt_box_mean() and guided_filter() size per-thread buffers by
  // dt_get_num_threads(), which clamps to this; dt_init() never runs here
  darktable.num_openmp_threads = dt_get_num_procs();
  return 0;
}

/*
 * TEST FUNCTIONS
 */

static void test_sanitize_nodes(void **state)
{
  for(int i = 0; i < LS_GOLDEN_NODES_BEFORE_N; i++)
  {
    float v[LS_LIGHT_NODES];
    memcpy(v, LS_GOLDEN_NODES_BEFORE[i], sizeof(v));
    _sanitize_nodes(v);
    for(int k = 0; k < LS_LIGHT_NODES; k++)
      assert_float_equal(v[k], LS_GOLDEN_NODES_AFTER[i][k], E);
  }
}

static void test_light_correction(void **state)
{
  for(int i = 0; i < LS_GOLDEN_CURVE_V_N; i++)
  {
    const float *const v = LS_GOLDEN_CURVE_V[i];
    float m[LS_LIGHT_NODES];
    _curve_tangents(v, m);

    gboolean zero = TRUE;
    for(int k = 0; k < LS_LIGHT_NODES; k++) zero = zero && v[k] == 0.0f;

    for(int j = 0; j < LS_GOLDEN_CURVE_X_N; j++)
    {
      const float C = _light_correction(LS_GOLDEN_CURVE_X[j], v, m);
      if(zero)
        assert_true(C == 0.0f);
      else
        assert_float_equal(C, LS_GOLDEN_CURVE_C[i][j], E);
    }
  }
}

static void test_light_delta(void **state)
{
  for(int i = 0; i < LS_GOLDEN_LIGHT_LAM_N; i++)
  {
    // the golden nodes are sanitized already, as commit_params() leaves them
    dt_iop_lightsurface_data_t d;
    _data_init(&d);
    d.form = LS_GOLDEN_LIGHT_FORM[i];
    d.relief = LS_GOLDEN_LIGHT_RELIEF[i];
    d.fill = LS_GOLDEN_LIGHT_FILL[i];
    memcpy(d.light_node, LS_GOLDEN_LIGHT_V[i], sizeof(d.light_node));
    float m[LS_LIGHT_NODES];
    _curve_tangents(d.light_node, m);

    const float lambda = LS_GOLDEN_LIGHT_LAM[i];
    const float lambda_f = LS_GOLDEN_LIGHT_LAM_F[i];
    const float dL = _light_delta(lambda, lambda - lambda_f, lambda_f, &d, m);

    if(_params_are_zero(&d))
      assert_true(dL == 0.0f);
    else
      assert_float_equal(dL, LS_GOLDEN_LIGHT_DL[i], E);
  }
}

static void test_hue_curve(void **state)
{
  for(int i = 0; i < LS_GOLDEN_HUECURVE_H_N; i++)
    assert_float_equal(_hue_curve(LS_GOLDEN_HUECURVE_H[i], LS_GOLDEN_HUECURVE_Q[i]),
                       LS_GOLDEN_HUECURVE_HOUT[i], E);

  for(int i = 0; i < LS_GOLDEN_HUECURVE_Q_N; i++)
  {
    const float *const q = LS_GOLDEN_HUECURVE_Q[i];

    // it passes through the nodes
    for(int j = 0; j < LS_HUE_NODES; j++)
      assert_true(_hue_curve(45.0f * j, q) == q[j]);

    // periodic, and continuous across 0/360 degrees
    const float h = LS_GOLDEN_HUECURVE_H[i];
    assert_float_equal(_hue_curve(h + 360.0f, q), _hue_curve(h, q), E);
    assert_float_equal(_hue_curve(h - 360.0f, q), _hue_curve(h, q), E);
    assert_float_equal(_hue_curve(360.0f - 1e-4f, q), _hue_curve(1e-4f, q), E);
  }

  // all-zero nodes give an exact zero
  const float q0[LS_HUE_NODES] = { 0.0f };
  for(int k = 0; k < 500; k++)
    assert_true(_hue_curve(_rand_uniform(0.0f, 360.0f), q0) == 0.0f);
}

static void test_surface_delta(void **state)
{
  for(int i = 0; i < LS_GOLDEN_SURFACE_Y_N; i++)
  {
    const float *const q = LS_GOLDEN_SURFACE_Q[i];
    const float dS = _surface_delta(LS_GOLDEN_SURFACE_Y[i], LS_GOLDEN_SURFACE_RHO[i],
                                    LS_GOLDEN_SURFACE_RHO_REF[i], LS_GOLDEN_SURFACE_SC[i],
                                    LS_GOLDEN_SURFACE_H[i], LS_GOLDEN_SURFACE_S[i], q);

    gboolean zero = LS_GOLDEN_SURFACE_SC[i] == 0.0f;
    for(int j = 0; j < LS_HUE_NODES; j++) zero = zero && q[j] == 0.0f;

    if(zero || LS_GOLDEN_SURFACE_Y[i] <= LS_EPS)
      assert_true(dS == 0.0f);
    else
      assert_float_equal(dS, LS_GOLDEN_SURFACE_DS[i], E);
  }
}

static void test_gain(void **state)
{
  // clamp first, then reject a NaN; compared exactly, the cases are exact
  for(int i = 0; i < LS_GOLDEN_GAIN_DL_N; i++)
    assert_true(_gain(LS_GOLDEN_GAIN_DL[i], LS_GOLDEN_GAIN_DS[i]) == LS_GOLDEN_GAIN_G[i]);

  assert_true(_gain(0.0f, 0.0f) == 0.0f);
}

static void test_hue_sat(void **state)
{
  // the golden hues use a plain Rec.2020 to XYZ D65 matrix, so the hue code
  // is compared on its own; real renders build darktable's D50 + CAT16 one
  dt_colormatrix_t M = { { 0.0f } };
  for(int r = 0; r < 3; r++)
    for(int c = 0; c < 3; c++)
      M[r][c] = LS_GOLDEN_RGB_TO_XYZ_D65[r][c];

  for(int i = 0; i < LS_GOLDEN_HUESAT_RGB_N; i++)
  {
    const dt_aligned_pixel_t rgb = { LS_GOLDEN_HUESAT_RGB[i][0], LS_GOLDEN_HUESAT_RGB[i][1],
                                     LS_GOLDEN_HUESAT_RGB[i][2], 0.0f };
    float h = 0.0f, s = 0.0f;
    _hue_sat(rgb, M, &h, &s);

    assert_float_equal(s, LS_GOLDEN_HUESAT_S[i], E);

    // at saturation 0 the hue is undefined: dt UCS U and V are rounding
    // noise around 0 there, and its weight in the surface delta is 0
    if(LS_GOLDEN_HUESAT_S[i] == 0.0f) continue;

    // degrees, compared on the circle. 1e-3 rather than 1e-5: the hue is
    // atan2 of U and V after about ten float32 steps of the dt UCS chain,
    // which numpy and C round in different orders, and one float32 step at
    // 360 degrees alone is 3e-5
    const float dh = fabsf(h - LS_GOLDEN_HUESAT_H[i]);
    assert_float_equal(fminf(dh, 360.0f - dh), 0.0f, 1e-3f);
  }
}

static void _check_2d(const float *const out, const float gold[][LS_GOLDEN_BOX_A_M])
{
  for(int y = 0; y < LS_GOLDEN_BOX_A_N; y++)
    for(int x = 0; x < LS_GOLDEN_BOX_A_M; x++)
      assert_float_equal(out[y * LS_GOLDEN_BOX_A_M + x], gold[y][x], E);
}

static void test_box_mean(void **state)
{
  // the gray guided filter relies on dt_box_mean() being the in-bounds mean
  const int h = LS_GOLDEN_BOX_A_N, w = LS_GOLDEN_BOX_A_M;
  float *const buf = dt_alloc_align_float((size_t)w * h);
  assert_non_null(buf);

  memcpy(buf, LS_GOLDEN_BOX_A, sizeof(float) * w * h);
  dt_box_mean(buf, h, w, 1, 1, 1);
  _check_2d(buf, LS_GOLDEN_BOX_R1);

  memcpy(buf, LS_GOLDEN_BOX_A, sizeof(float) * w * h);
  dt_box_mean(buf, h, w, 1, 3, 1);
  _check_2d(buf, LS_GOLDEN_BOX_R3);

  dt_free_align(buf);
}

static void test_gray_guided(void **state)
{
  const int h = LS_GOLDEN_BOX_A_N, w = LS_GOLDEN_BOX_A_M;
  float *const out = dt_alloc_align_float((size_t)w * h);
  assert_non_null(out);

  _gray_guided(&LS_GOLDEN_BOX_A[0][0], out, w, h, 2, 0.3f);
  _check_2d(out, LS_GOLDEN_GGF_R2_E03);

  _gray_guided(&LS_GOLDEN_BOX_A[0][0], out, w, h, 3, 0.1f);
  _check_2d(out, LS_GOLDEN_GGF_R3_E01);

  dt_free_align(out);
}

static void test_light_delta_identity(void **state)
{
  dt_iop_lightsurface_data_t d;
  _data_init(&d);
  float m[LS_LIGHT_NODES];
  _curve_tangents(d.light_node, m);

  // == rather than a bit compare, so -0.0f passes
  for(int k = 0; k < 100000; k++)
  {
    const float lambda = _rand_uniform(-30.0f, 30.0f);
    const float lambda_f = _rand_uniform(-3.0f, 3.0f);
    assert_true(_light_delta(lambda, lambda - lambda_f, lambda_f, &d, m) == 0.0f);
  }
}

static void test_process_identity(void **state)
{
  // a random 16x16 layer brought to a 64x64 ROI the way process() does it:
  // bilinear samples refined by the guided filter with the log guide, then
  // the per-pixel loop at default data. the output must equal the input
  const int W = 64, H = 64, lw = 16, lh = 16;
  const size_t npix = (size_t)W * H;

  _ls_layer_t L = { 0 };
  L.w = lw;
  L.h = lh;
  L.cov_x = L.cov_y = 1.0f;
  L.y_floor = 1e-4f;
  L.mp.guide_radius_factor = 2.0f;
  L.mp.guide_sqrt_eps = 0.1f;
  L.lambda = dt_alloc_align_float((size_t)lw * lh);
  L.lambda_f = dt_alloc_align_float((size_t)lw * lh);
  assert_non_null(L.lambda);
  assert_non_null(L.lambda_f);
  for(int k = 0; k < lw * lh; k++)
  {
    L.lambda[k] = _rand_uniform(-10.0f, 5.0f);
    L.lambda_f[k] = _rand_uniform(-0.6f, 0.6f);
  }

  const dt_iop_roi_t roi = { .x = 0, .y = 0, .width = W, .height = H, .scale = 1.0f };
  dt_dev_pixelpipe_iop_t piece = { 0 };
  piece.buf_in.width = W;
  piece.buf_in.height = H;

  float *const in = dt_alloc_align_float(npix * 4);
  float *const out = dt_alloc_align_float(npix * 4);
  float *const guide = dt_alloc_align_float(npix * 4);
  float *const lambda = dt_alloc_align_float(npix);
  float *const lambda_f = dt_alloc_align_float(npix);
  float *const dS = dt_alloc_align_float(npix);
  assert_non_null(in);
  assert_non_null(out);
  assert_non_null(guide);
  assert_non_null(lambda);
  assert_non_null(lambda_f);
  assert_non_null(dS);

  for(size_t k = 0; k < npix * 4; k++) in[k] = _rand_uniform(-0.1f, 4.0f);
  for(size_t k = 0; k < npix; k++) dS[k] = 0.0f;

  _log_guide(in, npix, L.y_floor, guide);
  _light_full(&L, guide, &roi, &piece, lambda, lambda_f);

  dt_iop_lightsurface_data_t d;
  _data_init(&d);
  float m[LS_LIGHT_NODES];
  _curve_tangents(d.light_node, m);

  // without a surface edit process() passes no dS at all
  memset(out, 0xff, npix * 4 * sizeof(float));
  _apply_gain(in, out, lambda, lambda_f, NULL, npix, &d, m);
  assert_memory_equal(out, in, npix * 4 * sizeof(float));

  memset(out, 0xff, npix * 4 * sizeof(float));
  _apply_gain(in, out, lambda, lambda_f, dS, npix, &d, m);
  assert_memory_equal(out, in, npix * 4 * sizeof(float));

  // and the same loop does edit once a param moves
  d.form = 0.5f;
  _apply_gain(in, out, lambda, lambda_f, NULL, npix, &d, m);
  assert_memory_not_equal(out, in, npix * 4 * sizeof(float));

  dt_free_align(in);
  dt_free_align(out);
  dt_free_align(guide);
  dt_free_align(lambda);
  dt_free_align(lambda_f);
  dt_free_align(dS);
  dt_free_align(L.lambda);
  dt_free_align(L.lambda_f);
}

/*
 * MAIN FUNCTION
 */
int main(int argc, char *argv[])
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_sanitize_nodes),
    cmocka_unit_test(test_light_correction),
    cmocka_unit_test(test_light_delta),
    cmocka_unit_test(test_hue_curve),
    cmocka_unit_test(test_surface_delta),
    cmocka_unit_test(test_gain),
    cmocka_unit_test(test_hue_sat),
    cmocka_unit_test(test_box_mean),
    cmocka_unit_test(test_gray_guided),
    cmocka_unit_test(test_light_delta_identity),
    cmocka_unit_test(test_process_identity)
  };

  return cmocka_run_group_tests(tests, group_setup, NULL);
}
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
