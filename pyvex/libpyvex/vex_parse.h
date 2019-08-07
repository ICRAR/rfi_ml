/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_VEX_PARSE_H_INCLUDED
# define YY_YY_VEX_PARSE_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    T_VEX_REV = 258,
    T_REF = 259,
    T_DEF = 260,
    T_ENDDEF = 261,
    T_SCAN = 262,
    T_ENDSCAN = 263,
    T_CHAN_DEF = 264,
    T_SAMPLE_RATE = 265,
    T_BITS_PER_SAMPLE = 266,
    T_SWITCHING_CYCLE = 267,
    T_START = 268,
    T_SOURCE = 269,
    T_MODE = 270,
    T_STATION = 271,
    T_DATA_TRANSFER = 272,
    T_ANTENNA_DIAM = 273,
    T_ANTENNA_NAME = 274,
    T_AXIS_OFFSET = 275,
    T_ANTENNA_MOTION = 276,
    T_POINTING_SECTOR = 277,
    T_AXIS_TYPE = 278,
    T_BBC_ASSIGN = 279,
    T_CLOCK_EARLY = 280,
    T_RECORD_TRANSPORT_TYPE = 281,
    T_ELECTRONICS_RACK_TYPE = 282,
    T_NUMBER_DRIVES = 283,
    T_HEADSTACK = 284,
    T_RECORD_DENSITY = 285,
    T_TAPE_LENGTH = 286,
    T_RECORDING_SYSTEM_ID = 287,
    T_TAPE_MOTION = 288,
    T_TAPE_CONTROL = 289,
    T_TAI_UTC = 290,
    T_A1_TAI = 291,
    T_EOP_REF_EPOCH = 292,
    T_NUM_EOP_POINTS = 293,
    T_EOP_INTERVAL = 294,
    T_UT1_UTC = 295,
    T_X_WOBBLE = 296,
    T_Y_WOBBLE = 297,
    T_NUT_REF_EPOCH = 298,
    T_NUM_NUT_POINTS = 299,
    T_NUT_INTERVAL = 300,
    T_DELTA_PSI = 301,
    T_DELTA_EPS = 302,
    T_NUT_MODEL = 303,
    T_EXPER_NUM = 304,
    T_EXPER_NAME = 305,
    T_EXPER_NOMINAL_START = 306,
    T_EXPER_NOMINAL_STOP = 307,
    T_PI_NAME = 308,
    T_PI_EMAIL = 309,
    T_CONTACT_NAME = 310,
    T_CONTACT_EMAIL = 311,
    T_SCHEDULER_NAME = 312,
    T_SCHEDULER_EMAIL = 313,
    T_TARGET_CORRELATOR = 314,
    T_EXPER_DESCRIPTION = 315,
    T_HEADSTACK_POS = 316,
    T_IF_DEF = 317,
    T_PASS_ORDER = 318,
    T_S2_GROUP_ORDER = 319,
    T_PHASE_CAL_DETECT = 320,
    T_TAPE_CHANGE = 321,
    T_NEW_SOURCE_COMMAND = 322,
    T_NEW_TAPE_SETUP = 323,
    T_SETUP_ALWAYS = 324,
    T_PARITY_CHECK = 325,
    T_TAPE_PREPASS = 326,
    T_PREOB_CAL = 327,
    T_MIDOB_CAL = 328,
    T_POSTOB_CAL = 329,
    T_HEADSTACK_MOTION = 330,
    T_PROCEDURE_NAME_PREFIX = 331,
    T_ROLL_REINIT_PERIOD = 332,
    T_ROLL_INC_PERIOD = 333,
    T_ROLL = 334,
    T_ROLL_DEF = 335,
    T_SEFD_MODEL = 336,
    T_SEFD = 337,
    T_SITE_TYPE = 338,
    T_SITE_NAME = 339,
    T_SITE_ID = 340,
    T_SITE_POSITION = 341,
    T_SITE_POSITION_EPOCH = 342,
    T_SITE_POSITION_REF = 343,
    T_SITE_VELOCITY = 344,
    T_HORIZON_MAP_AZ = 345,
    T_HORIZON_MAP_EL = 346,
    T_ZEN_ATMOS = 347,
    T_OCEAN_LOAD_VERT = 348,
    T_OCEAN_LOAD_HORIZ = 349,
    T_OCCUPATION_CODE = 350,
    T_INCLINATION = 351,
    T_ECCENTRICITY = 352,
    T_ARG_PERIGEE = 353,
    T_ASCENDING_NODE = 354,
    T_MEAN_ANOMALY = 355,
    T_SEMI_MAJOR_AXIS = 356,
    T_MEAN_MOTION = 357,
    T_ORBIT_EPOCH = 358,
    T_SOURCE_TYPE = 359,
    T_SOURCE_NAME = 360,
    T_IAU_NAME = 361,
    T_RA = 362,
    T_DEC = 363,
    T_SOURCE_POSITION_REF = 364,
    T_RA_RATE = 365,
    T_DEC_RATE = 366,
    T_SOURCE_POSITION_EPOCH = 367,
    T_REF_COORD_FRAME = 368,
    T_VELOCITY_WRT_LSR = 369,
    T_SOURCE_MODEL = 370,
    T_VSN = 371,
    T_FANIN_DEF = 372,
    T_FANOUT_DEF = 373,
    T_TRACK_FRAME_FORMAT = 374,
    T_DATA_MODULATION = 375,
    T_VLBA_FRMTR_SYS_TRK = 376,
    T_VLBA_TRNSPRT_SYS_TRK = 377,
    T_S2_RECORDING_MODE = 378,
    T_S2_DATA_SOURCE = 379,
    B_GLOBAL = 380,
    B_STATION = 381,
    B_MODE = 382,
    B_SCHED = 383,
    B_EXPER = 384,
    B_SCHEDULING_PARAMS = 385,
    B_PROCEDURES = 386,
    B_EOP = 387,
    B_FREQ = 388,
    B_CLOCK = 389,
    B_ANTENNA = 390,
    B_BBC = 391,
    B_CORR = 392,
    B_DAS = 393,
    B_HEAD_POS = 394,
    B_PASS_ORDER = 395,
    B_PHASE_CAL_DETECT = 396,
    B_ROLL = 397,
    B_IF = 398,
    B_SEFD = 399,
    B_SITE = 400,
    B_SOURCE = 401,
    B_TRACKS = 402,
    B_TAPELOG_OBS = 403,
    T_LITERAL = 404,
    T_NAME = 405,
    T_LINK = 406,
    T_ANGLE = 407,
    T_COMMENT = 408,
    T_COMMENT_TRAILING = 409
  };
#endif
/* Tokens.  */
#define T_VEX_REV 258
#define T_REF 259
#define T_DEF 260
#define T_ENDDEF 261
#define T_SCAN 262
#define T_ENDSCAN 263
#define T_CHAN_DEF 264
#define T_SAMPLE_RATE 265
#define T_BITS_PER_SAMPLE 266
#define T_SWITCHING_CYCLE 267
#define T_START 268
#define T_SOURCE 269
#define T_MODE 270
#define T_STATION 271
#define T_DATA_TRANSFER 272
#define T_ANTENNA_DIAM 273
#define T_ANTENNA_NAME 274
#define T_AXIS_OFFSET 275
#define T_ANTENNA_MOTION 276
#define T_POINTING_SECTOR 277
#define T_AXIS_TYPE 278
#define T_BBC_ASSIGN 279
#define T_CLOCK_EARLY 280
#define T_RECORD_TRANSPORT_TYPE 281
#define T_ELECTRONICS_RACK_TYPE 282
#define T_NUMBER_DRIVES 283
#define T_HEADSTACK 284
#define T_RECORD_DENSITY 285
#define T_TAPE_LENGTH 286
#define T_RECORDING_SYSTEM_ID 287
#define T_TAPE_MOTION 288
#define T_TAPE_CONTROL 289
#define T_TAI_UTC 290
#define T_A1_TAI 291
#define T_EOP_REF_EPOCH 292
#define T_NUM_EOP_POINTS 293
#define T_EOP_INTERVAL 294
#define T_UT1_UTC 295
#define T_X_WOBBLE 296
#define T_Y_WOBBLE 297
#define T_NUT_REF_EPOCH 298
#define T_NUM_NUT_POINTS 299
#define T_NUT_INTERVAL 300
#define T_DELTA_PSI 301
#define T_DELTA_EPS 302
#define T_NUT_MODEL 303
#define T_EXPER_NUM 304
#define T_EXPER_NAME 305
#define T_EXPER_NOMINAL_START 306
#define T_EXPER_NOMINAL_STOP 307
#define T_PI_NAME 308
#define T_PI_EMAIL 309
#define T_CONTACT_NAME 310
#define T_CONTACT_EMAIL 311
#define T_SCHEDULER_NAME 312
#define T_SCHEDULER_EMAIL 313
#define T_TARGET_CORRELATOR 314
#define T_EXPER_DESCRIPTION 315
#define T_HEADSTACK_POS 316
#define T_IF_DEF 317
#define T_PASS_ORDER 318
#define T_S2_GROUP_ORDER 319
#define T_PHASE_CAL_DETECT 320
#define T_TAPE_CHANGE 321
#define T_NEW_SOURCE_COMMAND 322
#define T_NEW_TAPE_SETUP 323
#define T_SETUP_ALWAYS 324
#define T_PARITY_CHECK 325
#define T_TAPE_PREPASS 326
#define T_PREOB_CAL 327
#define T_MIDOB_CAL 328
#define T_POSTOB_CAL 329
#define T_HEADSTACK_MOTION 330
#define T_PROCEDURE_NAME_PREFIX 331
#define T_ROLL_REINIT_PERIOD 332
#define T_ROLL_INC_PERIOD 333
#define T_ROLL 334
#define T_ROLL_DEF 335
#define T_SEFD_MODEL 336
#define T_SEFD 337
#define T_SITE_TYPE 338
#define T_SITE_NAME 339
#define T_SITE_ID 340
#define T_SITE_POSITION 341
#define T_SITE_POSITION_EPOCH 342
#define T_SITE_POSITION_REF 343
#define T_SITE_VELOCITY 344
#define T_HORIZON_MAP_AZ 345
#define T_HORIZON_MAP_EL 346
#define T_ZEN_ATMOS 347
#define T_OCEAN_LOAD_VERT 348
#define T_OCEAN_LOAD_HORIZ 349
#define T_OCCUPATION_CODE 350
#define T_INCLINATION 351
#define T_ECCENTRICITY 352
#define T_ARG_PERIGEE 353
#define T_ASCENDING_NODE 354
#define T_MEAN_ANOMALY 355
#define T_SEMI_MAJOR_AXIS 356
#define T_MEAN_MOTION 357
#define T_ORBIT_EPOCH 358
#define T_SOURCE_TYPE 359
#define T_SOURCE_NAME 360
#define T_IAU_NAME 361
#define T_RA 362
#define T_DEC 363
#define T_SOURCE_POSITION_REF 364
#define T_RA_RATE 365
#define T_DEC_RATE 366
#define T_SOURCE_POSITION_EPOCH 367
#define T_REF_COORD_FRAME 368
#define T_VELOCITY_WRT_LSR 369
#define T_SOURCE_MODEL 370
#define T_VSN 371
#define T_FANIN_DEF 372
#define T_FANOUT_DEF 373
#define T_TRACK_FRAME_FORMAT 374
#define T_DATA_MODULATION 375
#define T_VLBA_FRMTR_SYS_TRK 376
#define T_VLBA_TRNSPRT_SYS_TRK 377
#define T_S2_RECORDING_MODE 378
#define T_S2_DATA_SOURCE 379
#define B_GLOBAL 380
#define B_STATION 381
#define B_MODE 382
#define B_SCHED 383
#define B_EXPER 384
#define B_SCHEDULING_PARAMS 385
#define B_PROCEDURES 386
#define B_EOP 387
#define B_FREQ 388
#define B_CLOCK 389
#define B_ANTENNA 390
#define B_BBC 391
#define B_CORR 392
#define B_DAS 393
#define B_HEAD_POS 394
#define B_PASS_ORDER 395
#define B_PHASE_CAL_DETECT 396
#define B_ROLL 397
#define B_IF 398
#define B_SEFD 399
#define B_SITE 400
#define B_SOURCE 401
#define B_TRACKS 402
#define B_TAPELOG_OBS 403
#define T_LITERAL 404
#define T_NAME 405
#define T_LINK 406
#define T_ANGLE 407
#define T_COMMENT 408
#define T_COMMENT_TRAILING 409

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 17 "vex_parse.y" /* yacc.c:1909  */

int                     ival;
char                   *sval;
struct llist           *llptr;
struct qref            *qrptr;
struct def             *dfptr;
struct block           *blptr;
struct lowl            *lwptr;
struct dvalue          *dvptr;
struct external        *exptr;

struct chan_def        *cdptr;
struct switching_cycle *scptr;

struct station         *snptr;
struct data_transfer   *dtptr;

struct axis_type       *atptr;
struct antenna_motion  *amptr;
struct pointing_sector *psptr;

struct bbc_assign      *baptr;

struct headstack       *hsptr;

struct clock_early     *ceptr;

struct tape_length     *tlptr;
struct tape_motion     *tmptr;

struct headstack_pos   *hpptr;

struct if_def          *ifptr;

struct phase_cal_detect *pdptr;

struct setup_always    *saptr;
struct parity_check    *pcptr;
struct tape_prepass    *tpptr;
struct preob_cal       *prptr;
struct midob_cal       *miptr;
struct postob_cal      *poptr;

struct sefd            *septr;

struct site_position   *spptr;
struct site_velocity   *svptr;
struct ocean_load_vert *ovptr;
struct ocean_load_horiz *ohptr;

struct source_model    *smptr;

struct vsn             *vsptr;

struct fanin_def	*fiptr;
struct fanout_def	*foptr;
struct vlba_frmtr_sys_trk	*fsptr;
struct s2_data_source  *dsptr;


#line 423 "vex_parse.h" /* yacc.c:1909  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_VEX_PARSE_H_INCLUDED  */
