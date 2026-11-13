#pragma once

// Shared real spherical-harmonic basis evaluation (CPU + GPU).
//
// This single header is included by BOTH the C++ CPU projector
// (ImageBasedPBR.cpp) and the HLSL forward shader (SimpleForward.hlsl), so
// projection and reconstruction use a bit-for-bit identical basis.
//
// Convention (identical to SHE_Math.hlsli for the overlapping bands):
//   - z is the polar axis, Condon-Shortley phase, Sloan real-SH normalization.
//   - coefficient index k = l*l + l + m, with m = -l .. +l.
//   - l = 0 .. (SH_CLASSIC_MAX_ORDER-1), SH_CLASSIC_COEFF_COUNT = order^2.
// Order-10 SH therefore uses l = 0..9 (100 coefficients). Index 0..3 are
// Y(0,0), Y(1,-1), Y(1,0), Y(1,1), matching SHBasisFunction3/5 exactly.
//
// The basis is built with the associated-Legendre recursion and an
// angle-addition recurrence for cos(m*phi)/sin(m*phi): no trig calls, only
// sqrt. That keeps this valid, cheap HLSL 6.0 code. The basis is written into
// a caller-provided array via SHB_OUT so both languages share one call shape:
//   float basis[SH_CLASSIC_COEFF_COUNT]; SHBasisEval(x, y, z, basis);

#ifndef SH_CLASSIC_MAX_ORDER
#define SH_CLASSIC_MAX_ORDER 10
#endif
#ifndef SH_CLASSIC_COEFF_COUNT
#define SH_CLASSIC_COEFF_COUNT (SH_CLASSIC_MAX_ORDER * SH_CLASSIC_MAX_ORDER)
#endif
#ifndef SH_CLASSIC_MAX_BANDS
#define SH_CLASSIC_MAX_BANDS (SH_CLASSIC_MAX_ORDER - 1)
#endif

#ifdef __cplusplus
#include <math.h>
#define SHB_LOOP
#define SHB_OUT float *Y
#else
#define SHB_LOOP [loop]
#define SHB_OUT inout float Y[SH_CLASSIC_COEFF_COUNT]
#endif

// Fill Y[k] = Y_l^m(x,y,z) for k = l*l + l + m, l = 0 .. MAX_BANDS.
void SHBasisEval(float x, float y, float z, SHB_OUT)
{
	const int L = SH_CLASSIC_MAX_BANDS;
	const float shbPI = 3.14159265359f;
	const float shbSQRT2 = 1.41421356237f;

	SHB_LOOP
	for (int q = 0; q < SH_CLASSIC_COEFF_COUNT; ++q)
	{
		Y[q] = 0.0f;
	}

	// P(id(l,m)) = P_l^m(z), flattened 2D array (l * MAX + m).
	float P[SH_CLASSIC_MAX_ORDER * SH_CLASSIC_MAX_ORDER];
	SHB_LOOP
	for (int q = 0; q < SH_CLASSIC_MAX_ORDER * SH_CLASSIC_MAX_ORDER; ++q)
	{
		P[q] = 0.0f;
	}

#define SHB_P(l, m) P[(l) * SH_CLASSIC_MAX_ORDER + (m)]

	float s2 = 1.0f - z * z;
	float s = s2 > 0.0f ? (float)sqrt(s2) : 0.0f; // sin(theta)
	float cp = 1.0f, sp = 0.0f;            // cos(phi), sin(phi)
	if (s > 1e-6f)
	{
		cp = x / s;
		sp = y / s;
	}

	// cos(m*phi), sin(m*phi) via angle-addition recurrence.
	float cosM[SH_CLASSIC_MAX_ORDER];
	float sinM[SH_CLASSIC_MAX_ORDER];
	cosM[0] = 1.0f;
	sinM[0] = 0.0f;
	SHB_LOOP
	for (int m = 1; m <= L; ++m)
	{
		cosM[m] = cosM[m - 1] * cp - sinM[m - 1] * sp;
		sinM[m] = sinM[m - 1] * cp + cosM[m - 1] * sp;
	}

	// Associated Legendre recursion (Condon-Shortley phase included).
	SHB_P(0, 0) = 1.0f;
	SHB_LOOP
	for (int m = 1; m <= L; ++m)
	{
		SHB_P(m, m) = -(2.0f * float(m) - 1.0f) * s * SHB_P(m - 1, m - 1);
	}
	SHB_LOOP
	for (int m = 0; m <= L - 1; ++m)
	{
		SHB_P(m + 1, m) = (2.0f * float(m) + 1.0f) * z * SHB_P(m, m);
	}
	SHB_LOOP
	for (int m = 0; m <= L; ++m)
	{
		SHB_LOOP
		for (int l = m + 2; l <= L; ++l)
		{
			SHB_P(l, m) = ((2.0f * float(l) - 1.0f) * z * SHB_P(l - 1, m)
				- float(l + m - 1) * SHB_P(l - 2, m)) / float(l - m);
		}
	}

	SHB_LOOP
	for (int l = 0; l <= L; ++l)
	{
		SHB_LOOP
		for (int m = -l; m <= l; ++m)
		{
			int am = m < 0 ? -m : m;
			// K_l^|m| = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
			float f = (2.0f * float(l) + 1.0f) / (4.0f * shbPI);
			SHB_LOOP
			for (int k = l - am + 1; k <= l + am; ++k)
			{
				f /= float(k);
			}
			float K = (float)sqrt(f);
			float p = SHB_P(l, am);
			float val;
			if (m == 0)      val = K * p;
			else if (m > 0)  val = shbSQRT2 * K * cosM[m] * p;
			else             val = shbSQRT2 * K * sinM[am] * p;
			Y[l * l + l + m] = val;
		}
	}
#undef SHB_P
}
