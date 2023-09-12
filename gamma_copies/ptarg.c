#include "typedef_MSP.h"
#include <fftw3.h>

#define MAX(a,b)  ( ( (a) > (b) ) ? (a) : (b) )
#define MIN(a,b)  ( ( (a) < (b) ) ? (a) : (b) )

#define OSF		16		/* oversampling factor */
#define SZ		64		/* size of extracted region */
#define FILE_LDR 	0		/* size of file leader */
#define REC_LDR  	0		/* size of record leader/line */
#define TWO_PI    6.283185307179586
#define RTD	  57.2957795131		/* radians to degrees */
#define DTR	  .0174532925199	/* degrees to radians */
#define DBMIN	-60.0			/* lower limit for dB scale */
#define MAX_STR  1024			/* length of a Gnuplot title */

extern void gplot1(FILE *gp, char *term,int win, char *title, char *xlabel, char *ylabel, char *pngfn, double xmin, double xmax, double ymin, double ymax, double *x, double *y, int np);
extern void *calloc_1d(size_t nv, size_t size);
extern void **calloc_2d(size_t ncols, size_t nlines, size_t size);
extern void zero_1d(void *a, int width, size_t size);
extern void zero_2d(void **a, int width, int nlines, size_t size);

extern void start_timing();
extern void stop_timing();

int main(int argc, char **argv) {
  fcomplex **ptr, **ptr1, **ptr2, **ptr2_tmp; /* point target, oversampled point target arrays */
  fcomplex psr, psaz;			/* gradients in range and azimuth */
  fcomplex c1, c2;
  void *tmp;

  double **ptr3;		/* detected intensity */
  double *rp, *rpdb, *rph;	/* range cut power,power dB, phase */
  double *ap, *apdb, *aph;	/* azimuth cut power,power db, phase */
  double *rpos, *azpos;		/* range and azimuth position */
  double rpg, apg;		/* range and azimuth phase gradients */
  double pmx;			/* maximum power */
  double p1;			/* image intensity */
  double scf;			/* scale factor */
  double rdel, rpk;		/* range peak offset, range peak value */
  double adel, apk;		/* azimuth peak offset, azimuth peak value */
  double b0, b1, b2;		/* parabolic interpolation constants */
  double r3db, a3db;		/* range and azimuth -3dB width */
  double r10db, a10db;		/* range and azimuth -10 dB width */
  double rpslr, apslr;		/* range and azimuth PSLR */
  double rsle, rmle;		/* integrated range side lobe energy and main lobe energy */
  double asle, amle;		/* integrated azimuth side lobe energy and main lobe energy */
  double rislr, aislr;		/* integrated range and azimuth sidel lobe energy */
  double phg;			/* phase gradient */
  double ph0, ph1;		/* peak phase values */

  int width, nbyte, nb;		/* width of records, bytes/line, number of bytes to copy/line */
  int i, k, n, j, j1, k1;	/* loop counter */
  int kk, jj;			/* oversampled image position of peak */
  int rec_ldr = REC_LDR;	/* record leader size in bytes */
  int file_ldr = FILE_LDR;	/* file header length in bytes */
  int rsz;			/* size of range region to copy */
  int asz;			/* size of azimuth region to copy */
  int rctr, actr;		/* center coordinates (range, azimuth) */
  int kx, jx;			/* nearest integer location to maximum */
  int jn1, jn2, kn1, kn2;	/* indices of first range and azimuth nulls */
  int jpslr, kpslr;		/* indices of peak side lobes in range and azimuth */
  int nmbr, nmba;		/* number of bins in the main lobe, range and azimuth */
  int nlf;			/* number of lines in the file */
  int image_format = FCMPLX_DATA;/* format of the SLC or range compressed data */
  int bpp;			/* bytes/pixel */
  int iwin = 1;			/* search window radius (samples) */
  int pltflg = 0;		/* plot flag 0: none, 1: png, 2:wxy and png */

  off_t offset = 0;		/* offset in bytes to begin copying */
  fftwf_plan plan1, plan2;
  FILE *f1, *f2, *f3, *f4;

  printf("*** Point target response analysis and interpolation ***\n") ;
  printf("*** Copyright 2016, Gamma Remote Sensing, v2.3 19-Feb-2016 clw ***\n");
#ifdef WITH_LICENSE
  {
#include "package.h"
#ifdef __cplusplus
  extern int verifyLicense(char* packageName);
  verifyLicense((char *)PACKAGE);
#else
  extern int c_verifyLicense(char* packageName);
  c_verifyLicense((char *)PACKAGE);
#endif
  }
#endif
  if (argc < 9) {
    printf("\nusage: %s <SLC> <width> <r_samp> <az_samp> <ptr_image> <r_plot> <az_plot> <data_format> [win] [pltflg]\n\n", argv[0]) ;
    printf("input parameters:\n");
    printf("  SLC          (input) SLC in fcomplex or scomplex format\n");
    printf("  width        (input) SLC complex samples per line\n");
    printf("  r_samp       (input) point target range sample number\n");
    printf("  az_samp      (input) point target azimuth line number\n");
    printf("  ptr_image    (output) oversampled point target image (fcomplex, 1024x1024 samples), with and without phase gradient\n");
    printf("  r_plot       (output) range point target response plot data (text format)\n");
    printf("  az_plot      (output) azimuth point target response plot data (text format)\n");
    printf("  data_format  input data format flag (default:from MSP processing parameter file)\n");
    printf("                 0: FCOMPLEX (pairs of 4-byte float)\n");
    printf("                 1: SCOMPLEX (pairs of 2-byte short integer)\n");
    printf("  win          maximum search window offset (samples) (integer, default: 1)\n");
    printf("  pltflg       azimuth spectrum plotting flag:\n");
    printf("                 0: none (default)\n");
    printf("                 1: output plots in PNG format\n");
    printf("                 2: screen output and PNG format plots\n\n");
    exit(-1);
  }

  start_timing();
  f1 = fopen(argv[1], FOPEN_RDONLY_BINARY);
  if (f1 == NULL) {
    fprintf(stderr, "\nERROR: cannot open input SLC file: %s\n\n", argv[1]);
    exit(-1);
  }

  f2 = fopen(argv[5], FOPEN_WRONLY_BINARY);
  if (f2 == NULL) {
    fprintf(stderr, "\nERROR: cannot open output oversampled point target image file: %s\n\n", argv[5]);
    exit(-1);
  }

  f3 = fopen(argv[6], FOPEN_WRONLY_BINARY);
  if (f3 == NULL) {
    fprintf(stderr, "\nERROR: cannot open range cut output file: %s\n\n", argv[6]);
    exit(-1);
  }

  f4 = fopen(argv[7], FOPEN_WRONLY_BINARY);
  if (f4 == NULL) {
    fprintf(stderr, "\nERROR: cannot open azimuth cut output file: %s\n\n", argv[7]);
    exit(-1);
  }
  fflush(stderr);

  sscanf(argv[2], "%d", &width);
  sscanf(argv[3], "%d", &rctr);
  sscanf(argv[4], "%d", &actr);
  rsz = SZ;
  asz = SZ;				/* initialize range and azimuth window sizes */

  sscanf(argv[8], "%d", &image_format);
  if ((image_format != FCMPLX_DATA) && (image_format != SCMPLX_DATA)) {
    fprintf(stderr, "\nERROR: command line value for input data format is invalid: %d\n\n", image_format);
    exit(-1);
  }

  switch (image_format) {
  case FCMPLX_DATA:
    printf("input data format: FCOMPLEX\n");
    bpp = sizeof(fcomplex);
    break;
  case SCMPLX_DATA:
    printf("input data format: SCOMPLEX\n");
    bpp = sizeof(scomplex);
    break;
  default:
    fprintf(stderr, "\nERROR invalid input data format: %d\n\n", image_format);
    exit(-1);
  }

  fseek(f1, (off_t)0L, SEEK_END);
  nbyte = width * bpp + rec_ldr;	/* number of bytes/line */
  nb = rsz * bpp;			/* number of bytes to read */

  nlf = (int)(ftell(f1) / ((off_t)(nbyte)));
  rewind(f1);

  if ((actr - asz / 2) < 0) {
    fprintf(stderr, "\nERROR: starting line of extracted region < 0\n\n");
    exit(-1);
  }
  if ((rctr - rsz / 2) < 0) {
    fprintf(stderr, "\nERROR: starting sample of extracted region < 0\n\n");
    exit(-1);
  }
  if ((actr + asz / 2) > nlf - 1) {
    fprintf(stderr, "\nERROR: ending line of extracted region exceeds file length: %d\n\n", nlf);
    exit(-1);
  }
  if ((rctr + rsz / 2) > width - 1) {
    fprintf(stderr, "\nERROR: last range sample of exceeds width %d\n\n", width);
    exit(-1);
  }

  if(argc > 9){
    sscanf(argv[9], "%d",&iwin);
    if(iwin > SZ/4){
      fprintf(stderr,"\nERROR: maximum size for win parameter is: %d\n",SZ/4);
      exit(-1);
    }
  }

  printf("\ninput SLC data file: %s\n", argv[1]);
  printf("width (samples): %6d  lines: %6d\n", width, nlf);
  printf("file leader size (bytes):   %6d\n", file_ldr);
  printf("record leader size (bytes): %6d\n", rec_ldr);
  printf("number of samples/line:     %6d\n", width);
  printf("number of bytes/line        %6d  bytes/line extracted: %d\n", nbyte, nb);
  printf("analysis window size range: %d   azimuth: %d\n", rsz, asz);
  printf("estimated point target center (range,azimuth): %6d  %6d\n", rctr, actr);
  printf("output oversampled SLC data file:     %s\n", argv[5]);
  printf("output range point target response:   %s\n", argv[6]);
  printf("output azimuth point target response: %s\n\n", argv[7]);
  printf("maximum search window offset (samples): %d\n",iwin);
  fflush(stdout);

  ptr = (fcomplex **)calloc_2d(rsz, asz, sizeof(fcomplex *));
  ptr[0] = (fcomplex *)fftwf_malloc(sizeof(fcomplex) * asz * OSF * rsz * OSF);
  zero_1d(ptr[0], asz*OSF*rsz*OSF, sizeof(fcomplex));

  ptr1 = (fcomplex **)calloc_1d(OSF*asz, sizeof(fcomplex *));
  ptr1[0] = (fcomplex *)fftwf_malloc(sizeof(fcomplex) * asz * OSF * rsz * OSF);
  zero_1d(ptr1[0], asz*OSF*rsz*OSF, sizeof(fcomplex));

  ptr2 = (fcomplex **)calloc_1d(OSF*asz, sizeof(fcomplex *));
  ptr2[0] = (fcomplex *)fftwf_malloc(sizeof(fcomplex) * asz * OSF * rsz * OSF);
  zero_1d(ptr2[0], asz*OSF*rsz*OSF, sizeof(fcomplex));

  ptr2_tmp = (fcomplex **)calloc_1d(asz*OSF, sizeof(fcomplex *));
  ptr2_tmp[0] = (fcomplex *)fftwf_malloc(sizeof(fcomplex) * asz * OSF * rsz * OSF);
  zero_1d(ptr2_tmp[0], asz*OSF*rsz*OSF, sizeof(fcomplex));

  ptr3 = (double **)calloc_1d(asz*OSF, sizeof(double *));
  ptr3[0] = (double *)calloc_1d(asz * OSF * rsz * OSF, sizeof(double));
  zero_1d(ptr3[0], asz*OSF*rsz*OSF, sizeof(double));

  for (i=1; i < asz*OSF; i++) {
    ptr1[i] = ptr1[0] + i*rsz*OSF;
    ptr2[i] = ptr2[0] + i*rsz*OSF;
    ptr2_tmp[i] = ptr2_tmp[0] + i*rsz*OSF;
    ptr3[i] = ptr3[0] + i*rsz*OSF;
  }
  
  tmp = (void *)calloc_1d(bpp * width + rec_ldr, sizeof(char));
  rp =   (double *)calloc_1d(rsz * OSF, sizeof(double));
  rpdb = (double *)calloc_1d(rsz * OSF, sizeof(double));
  rph =  (double *)calloc_1d(rsz * OSF, sizeof(double));
  ap =   (double *)calloc_1d(rsz * OSF, sizeof(double));
  apdb = (double *)calloc_1d(rsz * OSF, sizeof(double));
  aph =  (double *)calloc_1d(rsz * OSF, sizeof(double));
  rpos = (double *)calloc_1d(rsz * OSF, sizeof(double));
  azpos = (double *)calloc_1d(asz * OSF, sizeof(double));

  offset = (off_t)file_ldr + (off_t)nbyte * (actr - asz / 2) + (off_t)(rec_ldr + (rctr - rsz / 2) * bpp);
  fseek(f1, (off_t)offset, SEEK_SET);

  for (i = 0; i < asz; i++) {
    fread((char *)tmp, bpp / 2, 2*rsz, f1);
    switch (image_format) {
    case FCMPLX_DATA:
      for (j = 0; j < rsz; j++) {
        ptr[i][j].re = ((fcomplex *)tmp)[j].re;
        ptr[i][j].im = ((fcomplex *)tmp)[j].im;
      }
      break;
    case SCMPLX_DATA:
      for (j = 0; j < rsz; j++) {
        ptr[i][j].re = ((scomplex *)tmp)[j].re;
        ptr[i][j].im = ((scomplex *)tmp)[j].im;
      }
      break;
    default:
      fprintf(stderr, "\nERROR invalid input image data format: %d\n\n", image_format);
      exit(-1);
    }

    offset += nbyte;
    fseek(f1, (off_t)offset, SEEK_SET);
  }

  /* evaluate range and azimuth phase  gradients */
  psr.re = 0.0;
  psr.im = 0.0;
  psaz.re = 0.0;
  psaz.im = 0.0;

  for (k = 1; k < asz; k++) {
    for (n = 1; n < rsz; n++) {
      psr.re += (ptr[k][n].re * ptr[k][n-1].re + ptr[k][n].im * ptr[k][n-1].im);
      psr.im += (ptr[k][n].im * ptr[k][n-1].re - ptr[k][n].re * ptr[k][n-1].im);
      psaz.re += (ptr[k][n].re * ptr[k-1][n].re + ptr[k][n].im * ptr[k-1][n].im);
      psaz.im += (ptr[k][n].im * ptr[k-1][n].re - ptr[k][n].re * ptr[k-1][n].im);
    }
  }

  rpg = atan2((double)psr.im, (double)psr.re);
  apg = atan2((double)psaz.im, (double)psaz.re);

  printf("range phase gradient (degrees/sample):   %10.5f\n", rpg*RTD);
  printf("azimuth phase gradient (degrees/sample): %10.5f\n", apg*RTD);

  for (k = 1; k < asz; k++) {		/* remove phase gradient */
    for (n = 1; n < rsz; n++) {
      c1 = ptr[k][n];
      c2.re = cos(n * rpg + k * apg); 	/* conjg. phase (range+azimuth) */
      c2.im = -sin(n * rpg + k * apg);
      ptr[k][n].re = c1.re * c2.re - c1.im * c2.im;
      ptr[k][n].im = c1.im * c2.re + c1.re * c2.im;
    }
  }

  for (j = 0; j < OSF*asz; j++) {	/* zero out output arrays */
    for (k = 0; k < OSF*rsz; k++) {
      ptr1[j][k].re = 0.0;
      ptr1[j][k].im = 0.0;
      ptr2[j][k].re = 0.0;
      ptr2[j][k].im = 0.0;
      ptr2_tmp[j][k].re = 0.0;
      ptr2_tmp[j][k].im = 0.0;
      ptr3[j][k] = 0.0;
    }
  }

  for (j = 0; j < asz; j++) {		/* over sample the input data */
    j1 = OSF * j;
    for (k = 0; k < rsz; k++) {
      k1 = OSF * k;
      ptr1[j1][k1] = ptr[j][k];
    }
  }
  printf("\nforward FFT azimuth: %d  range: %d \n", asz*OSF, rsz*OSF);

  plan1 = fftwf_plan_dft_2d(asz*OSF, rsz*OSF, (fftwf_complex *)ptr1[0], (fftwf_complex *)ptr1[0], FFTW_FORWARD, FFTW_ESTIMATE);
  plan2 = fftwf_plan_dft_2d(asz*OSF, rsz*OSF, (fftwf_complex *)ptr2[0], (fftwf_complex *)ptr2[0], FFTW_BACKWARD, FFTW_ESTIMATE);

  fftwf_execute(plan1);		/* 2D forward FFT, OSF*asz rows, OSF*rsz col. ptr1->ptr1 */
  scf = 1. / (asz * rsz);	/* 2D FFT scale factor */

  for (k = -asz / 2; k < asz / 2; k++) {
    if (k < 0)k1 = OSF * asz + k;
    else k1 = k;

    for (j = -rsz / 2; j < rsz / 2; j++) {
      if (j < 0)j1 = OSF * rsz + j;
      else j1 = j;

      ptr2[k1][j1].re = scf * ptr1[k1][j1].re;
      ptr2[k1][j1].im = scf * ptr1[k1][j1].im;
    }
  }

  printf("inverse FFT azimuth: %d  range: %d \n", asz*OSF, rsz*OSF);
  fftwf_execute(plan2);		/* 2D backward FFT, OSF*asz rows, OSF*rsz col. ptr2->ptr2 */

  for (k = 0; k < asz*OSF; k++) {
    for (j = 0; j < rsz*OSF; j++) {	/* copy array due to byte swapping during write */
      ptr2_tmp[k][j].re = ptr2[k][j].re;
      ptr2_tmp[k][j].im = ptr2[k][j].im;
    }
    if (fwrite((char *)(ptr2_tmp[k]), sizeof(float), 2*rsz*OSF, f2) != (int)(2*rsz*OSF)) {
      fprintf(stderr, "\nERROR: error in writing to file: %s\n\n", argv[5]);
      exit(-1);
    }
  }

  for (k = 0; k < asz*OSF; k++) {	/* add back phase gradient */
    for (j = 0; j < rsz*OSF; j++) {	/* copy array due to possible byte swapping during write */
      phg = rpg * (double)j / OSF + apg * (double)k / OSF;
      c1.re = cos(phg);
      c1.im = sin(phg);
      ptr2_tmp[k][j].re = ptr2[k][j].re * c1.re - ptr2[k][j].im * c1.im;
      ptr2_tmp[k][j].im = ptr2[k][j].im * c1.re + ptr2[k][j].re * c1.im;
    }
    if (fwrite((char *)(ptr2_tmp[k]), sizeof(float), 2*rsz*OSF, f2) != (int)(2*rsz*OSF)) {
      fprintf(stderr, "\nERROR: error in writing to file: %s\n\n", argv[5]);
      exit(-1);
    }
  }

  for (k = 0; k < asz*OSF; k++) {	/* detect interpolated image */
    for (j = 0; j < rsz*OSF; j++) {
      c1 = ptr2[k][j];
      ptr3[k][j] = c1.re * c1.re + c1.im * c1.im;
    }
  }

  pmx = 0.0;
  kx = 0;
  jx = 0;
  for (k = 0; k < 2*iwin*OSF; k++) {
    kk = (asz * OSF) / 2 - iwin*OSF + k;

    for (j = 0; j < 2*iwin*OSF; j++) {
      jj = (rsz * OSF) / 2 - iwin*OSF + j;

      p1 = ptr3[kk][jj];
      if (p1 > pmx) {
        pmx = p1;
        kx = kk;
        jx = jj;
      }
    }
  }
  printf("\noversampled SLC analysis window peak value (nearest neighbor): %12.5e\n", pmx);
  printf("position in analysis window: range sample: %d  azimuth sample: %d\n", jx, kx);

  b0 =  ptr3[kx][jx];
  b1 = (ptr3[kx][jx+1] - ptr3[kx][jx-1]) / 2.;
  b2 = (ptr3[kx][jx+1] + ptr3[kx][jx-1] - 2. * b0) / 2.;
  rdel = (-b1 / (2. * b2));
  rpk = b0 + rdel * (b1 + rdel * b2);
  printf("\nSLC range_peak_position:   %10.5f  value: %12.5e\n", rctr + (rdel + jx - OSF*rsz / 2.) / OSF, rpk);

  b0 =  ptr3[kx][jx];
  b1 =  (ptr3[kx+1][jx] - ptr3[kx-1][jx]) / 2.;
  b2 =  (ptr3[kx+1][jx] + ptr3[kx-1][jx] - 2. * b0) / 2.;
  adel = (-b1 / (2. * b2));
  apk = b0 + adel * (b1 + adel * b2);
  printf("SLC azimuth_peak_position: %10.5f  value: %12.5e\n", actr + (adel + kx - OSF*asz / 2.) / OSF, apk);

  pmx = (rpk + apk) / 2.0;
  printf("SLC interpolated_peak_power_value: %12.5e\n\n", pmx);

  c1 = ptr2[kx][jx];		/* assume very low residual phase gradient */
  ph0 = atan2((double)c1.im, (double)c1.re);
  ph1 = ph0 + rpg * (double)(jx + rdel) / OSF + apg * (double)(kx + adel) / OSF;
  printf("SLC interpolated peak phase (no phase gradients) (rad): %11.4f\n", ph0);
  printf("SLC interpolated peak phase (with phase gradients (rad):%11.4f\n", ph1);

  for (k = 0; k < asz*OSF; k++) {
    for (j = 0; j < rsz*OSF; j++) {
      ptr2[k][j].re /= sqrt(pmx);
      ptr2[k][j].im /= sqrt(pmx);
      ptr3[k][j] /= pmx;
    }
  }

  for (j = 0; j < rsz*OSF; j++) {	/* range point target response */
    c1 = ptr2[kx][j];
    rp[j] = c1.re * c1.re + c1.im * c1.im;
    if (rp[j] > 0.0)rph[j] = atan2((double)c1.im, (double)c1.re);
    else rph[j] = 0.0;
    if (rp[j] > .000001)rpdb[j] = 10. * log10(rp[j]);
    else rpdb[j] = DBMIN;
    /* fprintf(f3,"%10.5f %10.4f %10.4f %10.4f\n",(j- OSF*rsz/2.)/OSF,rpdb[j],rp[j],rph[j]); */
    phg = rpg * (double)j / OSF + apg * (double)kx / OSF;
    rpos[j] = (j - jx - rdel) / OSF;
    fprintf(f3, "%10.5f %10.4f %10.4f %10.4f\n",rpos[j], rpdb[j], rph[j], rph[j] + phg);
  }

  for (j = jx; j < rsz*OSF; j++) {	/* -3 dB range resolution */
    if (rp[j] < .5) break;
  }

  b1 = rp[j] - rp[j-1];			/* linear interpolation */
  b0 = rp[j-1] - b1 * (j - 1);
  r3db = 2. * (.5 / b1 - b0 / b1 - jx) / OSF;

  for (j = jx; j < rsz*OSF; j++) {	/* -10 dB range resolution */
    if (rp[j] < .1) break;
  }
  b1 = rp[j] - rp[j-1];
  b0 = rp[j-1] - b1 * (j - 1);
  r10db = 2. * (.1 / b1 - b0 / b1 - jx) / OSF;

  /* range PSLR */
  for (j = jx - 1; j > 0; j--) {	/* search for first null to the left */
    if (rp[j] > rp[j+1]) break;
  }
  jn1 = j;
  for (j = jx + 1; j < rsz*OSF; j++) {	/* search for first null to the right */
    if (rp[j] > rp[j-1]) break;
  }
  jn2 = j;

  rpslr = rp[jn1];
  jpslr = jn1;
  for (j = jn1; j > 0; j--) {		/* search for maximum peak sidelobe to the left */
    if (rp[j] > rpslr) {
      rpslr = rp[j];
      jpslr = j;
    }
  }
  for (j = jn2; j < rsz*OSF; j++) {	/* search for maximum peak sidelobe to the right */
    if (rp[j] > rpslr) {
      rpslr = rp[j];
      jpslr = j;
    }
  }
  rpslr = rpdb[jpslr];

  printf("range -3 dB width (samples):    %10.3f\n", r3db);
  printf("range -10 dB width (samples):   %10.3f\n", r10db);
  printf("range PSLR (dB):                %10.3f\n", rpslr);
  /* calculate ISLR */
  nmbr = jn2 - jn1 + 1;
  for (j = jn1, rmle = 0.0; j <= jn2; j++) {
    rmle += rp[j];  /* peak integrated power */
  }
  if ((jn2 + 2*nmbr) > (rsz*OSF)) {
    printf("\nWARNING: point target not range centered in the analysis window, range ISLR cannot be determined.\n\n");
    goto azimuth;
  }
  for (j = jn2 + 1, rsle = 0.0; j < jn2 + 2*nmbr; j++) {
    rsle += rp[j];  			/* side lobe integrated power */
  }
  for (j = jn1 - 1; j > jn1 - 2*nmbr; j--) {
    rsle += rp[j];
  }
  rislr = 10. * log10(rsle / rmle);
  printf("range ISLR (dB):                %10.3f\n", rislr);

azimuth:
  for (k = 0; k < asz*OSF; k++) {	/* azimuth point target response */
    c1 = ptr2[k][jx];
    ap[k] = c1.re * c1.re + c1.im * c1.im;

    if (ap[k] > 0.0)aph[k] = atan2((double)c1.im, (double)c1.re);
    else aph[k] = 0.0;

    if (ap[k] > 1.e-20)apdb[k] = 10. * log10(ap[k]);
    else apdb[k] = DBMIN;

    phg = rpg * (double)jx / OSF + apg * (double)k / OSF;
    azpos[k] = (k - kx - adel) / OSF;
    fprintf(f4, "%10.5f %10.4f %10.4f %10.4f\n", azpos[k], apdb[k], aph[k], aph[k] + phg);
  }

  for (k = kx; k < asz*OSF; k++) {
    if (ap[k] < .5) break; 		/* -3 dB azimuth resolution */
  }
  b1 = ap[k] - ap[k-1];			/* linear interpolation */
  b0 = ap[k-1] - b1 * (k - 1);
  a3db = 2. * (.5 / b1 - b0 / b1 - kx) / OSF;

  for (k = kx; k < asz*OSF; k++) {
    if (ap[k] < .1) break;  		/* -10 dB azimuth resolution */
  }
  b1 = ap[k] - ap[k-1];
  b0 = ap[k-1] - b1 * (k - 1);
  a10db = 2. * (.1 / b1 - b0 / b1 - kx) / OSF;
  /* azimuth PSLR */
  for (k = kx - 1; k > 0; k--) {
    if (ap[k] > ap[k+1]) break;  	/* search for first null to the left */
  }
  kn1 = k;
  for (k = kx + 1; k < asz*OSF; k++) {
    if (ap[k] > ap[k-1]) break;  	/* search for first null to the right */
  }
  kn2 = k;

  apslr = ap[kn2];
  kpslr = kn1;
  for (k = kn1; k > 0; k--) {		/* search for maximum peak sidelobe to the left */
    if (ap[k] > apslr) {
      apslr = ap[k];
      kpslr = k;
    }
  }
  for (k = kn2; k < asz*OSF; k++) {	/* search for maximum peak sidelobe to the right */
    if (ap[k] > apslr) {
      apslr = ap[k];
      kpslr = k;
    }
  }
  apslr = apdb[kpslr];

  printf("\nazimuth -3 dB width  (samples): %10.3f\n", a3db);
  printf("azimuth -10 dB width (samples): %10.3f\n", a10db);
  printf("azimuth PSLR (dB):              %10.3f\n", apslr);
  nmba = kn2 - kn1 + 1;
  for (k = kn1, amle = 0.0; k <= kn2; k++) {
    amle += ap[k];  /* peak integrated power */
  }
  if ((kn2 + 2*nmba) > (asz*OSF)) {
    printf("\nWARNING: point target not azimuth centered in the analysis window, ISLR cannot be determined.\n");
    goto plot_data;
  }
  for (k = kn2 + 1, asle = 0.0; k < kn2 + 2*nmba; k++) {
    asle += ap[k];  /* side lobe integrated power */
  }
  for (k = kn1 - 1; k > kn1 - 2*nmba; k--) {
    asle += ap[k];
  }
  aislr = 10. * log10(asle / amle);
  printf("azimuth ISLR (dB): %10.3f\n", aislr);

plot_data:
  if ((argc > 10) && (strcmp(argv[10], "-") != 0)){
    sscanf(argv[10], "%d", &pltflg);
  }
  
#define GNUPLOT "gnuplot"
  if(pltflg != 0){
    char *ptitle1,*ptitle2, *pngfn1, *pngfn2;
    char *gt, *gt2, *str;
    FILE *gp;
    
    gp = popen(GNUPLOT, "w"); /* 'gp' is the pipe descriptor */
    if (gp==NULL) {
     printf("Error opening pipe to gnuplot. Check if you have installed gnuplot and if it is in the current search path\n");
     exit(-1);
    }
    
    pngfn1 = (char *)calloc_1d(strlen(argv[6])+32, sizeof(char));
    ptitle1 = (char *)calloc_1d(strlen(argv[6])+32, sizeof(char));
    gt = (char *)calloc_1d(MAX_STR, sizeof(char));
    str = (char *)calloc_1d(MAX_STR,sizeof(char));
    sprintf(ptitle1,"Range Point Target Response\\n%s",argv[6]);
    sprintf(pngfn1,"%s.png",argv[6]);
    printf("range point target Response: %s\n",pngfn1);

    pngfn2 = (char *)calloc_1d(strlen(argv[7])+32, sizeof(char));
    ptitle2 = (char *)calloc_1d(strlen(argv[7])+32, sizeof(char));
    sprintf(ptitle2,"Azimuth Point Target Response\\n%s",argv[7]);
    sprintf(pngfn2,"%s.png",argv[7]);
    printf("azimuth point target Response: %s\n",pngfn2);

    switch (pltflg){
    case 1:
      gplot1(gp,"png",0,ptitle1,"Range Sample","Relative Power (dB)",pngfn1,0.,0.,-40,3.,rpos,rpdb,rsz*OSF);
      gplot1(gp,"png",0,ptitle2,"Azimuth Sample","Relative Power (dB)",pngfn2,0.,0.,-40,3.,azpos,apdb,asz*OSF);
      break;

    case 2:
#if (defined __APPLE__ || defined __WIN32__)
      strncpy(gt,"wxt", MAX_STR - 1);
#endif
#ifdef __linux__
      strncpy(gt,"qt", MAX_STR - 1);
#endif
      gt2 = getenv ("GNUTERM");
      if(gt2 != NULL){
        printf("GNUTERM environment variable defined: %s\n",gt2);
        strncpy(gt, gt2, MAX_STR - 1);
      }

      gplot1(gp,"png",0,ptitle1,"Range Sample","Relative Power (dB)",pngfn1,0.,0.,-40,3.,rpos,rpdb,rsz*OSF);
      gplot1(gp,"png",0,ptitle2,"Azimuth Sample","Relative Power (dB)",pngfn2,0.,0.,-40,3.,azpos,apdb,asz*OSF);

      gplot1(gp,gt,0,ptitle1,"Range Sample","Relative Power (dB)",pngfn1,0.,0.,-40,3.,rpos,rpdb,rsz*OSF);
      gplot1(gp,gt,1,ptitle2,"Azimuth Sample","Relative Power (dB)",pngfn2,0.,0.,-40,3.,azpos,apdb,asz*OSF);
      printf("enter return to continue: ");
      fgets(str,2,stdin);

      break;
    default:
      fprintf(stderr,"\nERROR: invalid value for plot file flag: %d\n",pltflg);
      exit(-1);
    }
    fclose(gp);
  }

  stop_timing();
  return(0);
}
