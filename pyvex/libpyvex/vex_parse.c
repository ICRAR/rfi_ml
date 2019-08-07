/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "vex_parse.y" /* yacc.c:339  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vex.h"

#define YYDEBUG 1

/* globals */

struct vex *vex_ptr=NULL;
extern int lines;

#line 81 "vex_parse.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
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
#line 17 "vex_parse.y" /* yacc.c:355  */

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


#line 490 "vex_parse.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_VEX_PARSE_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 507 "vex_parse.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1467

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  159
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  282
/* YYNRULES -- Number of rules.  */
#define YYNRULES  675
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  1411

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   409

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,   158,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,   157,   156,
       2,   155,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   342,   342,   343,   345,   346,   348,   349,   350,   354,
     359,   360,   362,   363,   364,   365,   366,   367,   368,   369,
     370,   371,   372,   373,   374,   375,   376,   377,   378,   379,
     381,   382,   383,   384,   385,   389,   390,   394,   395,   397,
     398,   400,   401,   402,   404,   405,   409,   410,   412,   413,
     415,   416,   417,   419,   420,   425,   426,   428,   429,   430,
     432,   434,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   444,   445,   446,   447,   448,   449,   450,   451,   452,
     453,   455,   456,   458,   459,   460,   462,   463,   465,   466,
     470,   471,   473,   474,   476,   477,   478,   480,   482,   484,
     485,   487,   488,   489,   490,   491,   492,   493,   495,   497,
     499,   501,   510,   517,   524,   525,   527,   528,   530,   531,
     533,   534,   535,   537,   538,   540,   541,   543,   544,   546,
     547,   549,   550,   555,   556,   558,   559,   561,   562,   563,
     565,   567,   569,   570,   572,   573,   574,   575,   576,   577,
     578,   579,   580,   582,   584,   586,   589,   591,   596,   607,
     608,   610,   611,   613,   614,   615,   617,   618,   621,   622,
     624,   625,   626,   627,   629,   634,   635,   637,   638,   640,
     641,   642,   644,   646,   649,   650,   652,   653,   654,   655,
     657,   659,   661,   663,   668,   669,   671,   672,   674,   675,
     676,   678,   679,   682,   683,   685,   686,   687,   688,   689,
     690,   691,   693,   694,   695,   696,   697,   699,   701,   703,
     705,   707,   710,   713,   715,   718,   720,   722,   724,   728,
     732,   733,   735,   736,   738,   739,   740,   742,   743,   746,
     747,   749,   750,   751,   752,   753,   754,   755,   756,   757,
     758,   759,   760,   761,   762,   763,   764,   765,   767,   769,
     771,   773,   775,   777,   778,   780,   781,   783,   784,   786,
     788,   790,   792,   793,   795,   796,   798,   802,   803,   805,
     806,   808,   809,   810,   812,   814,   816,   817,   819,   820,
     821,   822,   824,   826,   827,   828,   829,   830,   831,   832,
     834,   835,   836,   838,   840,   842,   844,   846,   848,   850,
     852,   854,   856,   858,   860,   864,   865,   867,   868,   870,
     871,   872,   874,   875,   878,   879,   881,   882,   883,   884,
     885,   886,   887,   889,   897,   905,   913,   922,   923,   925,
     927,   929,   931,   936,   937,   939,   940,   942,   943,   944,
     946,   948,   951,   952,   954,   955,   956,   957,   959,   964,
     965,   967,   968,   970,   971,   972,   974,   975,   978,   979,
     981,   982,   983,   984,  1001,  1003,  1005,  1007,  1009,  1011,
    1013,  1020,  1021,  1023,  1024,  1027,  1028,  1029,  1031,  1033,
    1036,  1038,  1040,  1041,  1043,  1044,  1045,  1047,  1049,  1053,
    1054,  1056,  1058,  1060,  1061,  1062,  1064,  1066,  1068,  1070,
    1072,  1073,  1074,  1075,  1077,  1079,  1084,  1085,  1087,  1089,
    1091,  1092,  1093,  1095,  1097,  1100,  1102,  1104,  1106,  1108,
    1110,  1112,  1114,  1116,  1118,  1120,  1122,  1124,  1126,  1127,
    1128,  1130,  1132,  1134,  1136,  1138,  1141,  1144,  1147,  1150,
    1153,  1156,  1160,  1161,  1163,  1164,  1166,  1167,  1168,  1170,
    1172,  1175,  1176,  1178,  1179,  1180,  1181,  1182,  1183,  1184,
    1186,  1188,  1190,  1192,  1196,  1198,  1200,  1202,  1205,  1206,
    1207,  1209,  1211,  1214,  1216,  1219,  1220,  1221,  1222,  1226,
    1227,  1229,  1230,  1232,  1233,  1234,  1236,  1238,  1241,  1242,
    1244,  1245,  1246,  1247,  1248,  1250,  1252,  1257,  1258,  1260,
    1261,  1263,  1264,  1265,  1267,  1269,  1271,  1272,  1274,  1275,
    1276,  1277,  1278,  1279,  1280,  1281,  1282,  1283,  1284,  1285,
    1286,  1287,  1288,  1289,  1290,  1291,  1292,  1293,  1294,  1295,
    1296,  1297,  1299,  1301,  1303,  1305,  1309,  1311,  1313,  1317,
    1319,  1321,  1323,  1327,  1331,  1333,  1335,  1337,  1339,  1341,
    1343,  1345,  1347,  1351,  1352,  1354,  1355,  1357,  1358,  1359,
    1361,  1363,  1366,  1367,  1369,  1370,  1371,  1372,  1373,  1374,
    1375,  1376,  1377,  1378,  1379,  1380,  1381,  1382,  1383,  1384,
    1385,  1386,  1387,  1388,  1389,  1390,  1391,  1393,  1394,  1397,
    1399,  1401,  1403,  1405,  1407,  1409,  1411,  1413,  1415,  1418,
    1430,  1431,  1433,  1435,  1437,  1438,  1439,  1442,  1444,  1447,
    1449,  1451,  1452,  1453,  1454,  1457,  1462,  1463,  1465,  1466,
    1468,  1469,  1470,  1472,  1474,  1477,  1478,  1480,  1481,  1482,
    1484,  1485,  1487,  1489,  1490,  1491,  1492,  1493,  1495,  1498,
    1501,  1505,  1507,  1509,  1512,  1516,  1519,  1521,  1523,  1526,
    1528,  1533,  1536,  1538,  1539,  1541,  1542,  1544,  1545,  1547,
    1549,  1550,  1552,  1554,  1555,  1557
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "T_VEX_REV", "T_REF", "T_DEF",
  "T_ENDDEF", "T_SCAN", "T_ENDSCAN", "T_CHAN_DEF", "T_SAMPLE_RATE",
  "T_BITS_PER_SAMPLE", "T_SWITCHING_CYCLE", "T_START", "T_SOURCE",
  "T_MODE", "T_STATION", "T_DATA_TRANSFER", "T_ANTENNA_DIAM",
  "T_ANTENNA_NAME", "T_AXIS_OFFSET", "T_ANTENNA_MOTION",
  "T_POINTING_SECTOR", "T_AXIS_TYPE", "T_BBC_ASSIGN", "T_CLOCK_EARLY",
  "T_RECORD_TRANSPORT_TYPE", "T_ELECTRONICS_RACK_TYPE", "T_NUMBER_DRIVES",
  "T_HEADSTACK", "T_RECORD_DENSITY", "T_TAPE_LENGTH",
  "T_RECORDING_SYSTEM_ID", "T_TAPE_MOTION", "T_TAPE_CONTROL", "T_TAI_UTC",
  "T_A1_TAI", "T_EOP_REF_EPOCH", "T_NUM_EOP_POINTS", "T_EOP_INTERVAL",
  "T_UT1_UTC", "T_X_WOBBLE", "T_Y_WOBBLE", "T_NUT_REF_EPOCH",
  "T_NUM_NUT_POINTS", "T_NUT_INTERVAL", "T_DELTA_PSI", "T_DELTA_EPS",
  "T_NUT_MODEL", "T_EXPER_NUM", "T_EXPER_NAME", "T_EXPER_NOMINAL_START",
  "T_EXPER_NOMINAL_STOP", "T_PI_NAME", "T_PI_EMAIL", "T_CONTACT_NAME",
  "T_CONTACT_EMAIL", "T_SCHEDULER_NAME", "T_SCHEDULER_EMAIL",
  "T_TARGET_CORRELATOR", "T_EXPER_DESCRIPTION", "T_HEADSTACK_POS",
  "T_IF_DEF", "T_PASS_ORDER", "T_S2_GROUP_ORDER", "T_PHASE_CAL_DETECT",
  "T_TAPE_CHANGE", "T_NEW_SOURCE_COMMAND", "T_NEW_TAPE_SETUP",
  "T_SETUP_ALWAYS", "T_PARITY_CHECK", "T_TAPE_PREPASS", "T_PREOB_CAL",
  "T_MIDOB_CAL", "T_POSTOB_CAL", "T_HEADSTACK_MOTION",
  "T_PROCEDURE_NAME_PREFIX", "T_ROLL_REINIT_PERIOD", "T_ROLL_INC_PERIOD",
  "T_ROLL", "T_ROLL_DEF", "T_SEFD_MODEL", "T_SEFD", "T_SITE_TYPE",
  "T_SITE_NAME", "T_SITE_ID", "T_SITE_POSITION", "T_SITE_POSITION_EPOCH",
  "T_SITE_POSITION_REF", "T_SITE_VELOCITY", "T_HORIZON_MAP_AZ",
  "T_HORIZON_MAP_EL", "T_ZEN_ATMOS", "T_OCEAN_LOAD_VERT",
  "T_OCEAN_LOAD_HORIZ", "T_OCCUPATION_CODE", "T_INCLINATION",
  "T_ECCENTRICITY", "T_ARG_PERIGEE", "T_ASCENDING_NODE", "T_MEAN_ANOMALY",
  "T_SEMI_MAJOR_AXIS", "T_MEAN_MOTION", "T_ORBIT_EPOCH", "T_SOURCE_TYPE",
  "T_SOURCE_NAME", "T_IAU_NAME", "T_RA", "T_DEC", "T_SOURCE_POSITION_REF",
  "T_RA_RATE", "T_DEC_RATE", "T_SOURCE_POSITION_EPOCH",
  "T_REF_COORD_FRAME", "T_VELOCITY_WRT_LSR", "T_SOURCE_MODEL", "T_VSN",
  "T_FANIN_DEF", "T_FANOUT_DEF", "T_TRACK_FRAME_FORMAT",
  "T_DATA_MODULATION", "T_VLBA_FRMTR_SYS_TRK", "T_VLBA_TRNSPRT_SYS_TRK",
  "T_S2_RECORDING_MODE", "T_S2_DATA_SOURCE", "B_GLOBAL", "B_STATION",
  "B_MODE", "B_SCHED", "B_EXPER", "B_SCHEDULING_PARAMS", "B_PROCEDURES",
  "B_EOP", "B_FREQ", "B_CLOCK", "B_ANTENNA", "B_BBC", "B_CORR", "B_DAS",
  "B_HEAD_POS", "B_PASS_ORDER", "B_PHASE_CAL_DETECT", "B_ROLL", "B_IF",
  "B_SEFD", "B_SITE", "B_SOURCE", "B_TRACKS", "B_TAPELOG_OBS", "T_LITERAL",
  "T_NAME", "T_LINK", "T_ANGLE", "T_COMMENT", "T_COMMENT_TRAILING", "'='",
  "';'", "':'", "'*'", "$accept", "vex", "version_lowls", "version_lowl",
  "version", "blocks", "block", "global_block", "station_block",
  "station_defs", "station_defx", "station_def", "mode_block", "mode_defs",
  "mode_defx", "mode_def", "refs", "refx", "ref", "primitive", "qrefs",
  "qrefx", "qref", "qualifiers", "sched_block", "sched_defs", "sched_defx",
  "sched_def", "sched_lowls", "sched_lowl", "start", "mode", "source",
  "station", "data_transfer", "start_position", "pass", "sector", "drives",
  "scan_id", "method", "destination", "unit_value2", "options",
  "antenna_block", "antenna_defs", "antenna_defx", "antenna_def",
  "antenna_lowls", "antenna_lowl", "antenna_diam", "antenna_name",
  "axis_type", "axis_offset", "antenna_motion", "pointing_sector",
  "bbc_block", "bbc_defs", "bbc_defx", "bbc_def", "bbc_lowls", "bbc_lowl",
  "bbc_assign", "clock_block", "clock_defs", "clock_defx", "clock_def",
  "clock_lowls", "clock_lowl", "clock_early", "das_block", "das_defs",
  "das_defx", "das_def", "das_lowls", "das_lowl", "record_transport_type",
  "electronics_rack_type", "number_drives", "headstack", "record_density",
  "tape_length", "recording_system_id", "tape_motion", "tape_control",
  "eop_block", "eop_defs", "eop_defx", "eop_def", "eop_lowls", "eop_lowl",
  "tai_utc", "a1_tai", "eop_ref_epoch", "num_eop_points", "eop_interval",
  "ut1_utc", "x_wobble", "y_wobble", "nut_ref_epoch", "num_nut_points",
  "nut_interval", "delta_psi", "delta_eps", "nut_model", "exper_block",
  "exper_defs", "exper_defx", "exper_def", "exper_lowls", "exper_lowl",
  "exper_num", "exper_name", "exper_description", "exper_nominal_start",
  "exper_nominal_stop", "pi_name", "pi_email", "contact_name",
  "contact_email", "scheduler_name", "scheduler_email",
  "target_correlator", "freq_block", "freq_defs", "freq_defx", "freq_def",
  "freq_lowls", "freq_lowl", "chan_def", "switch_states", "switch_state",
  "sample_rate", "bits_per_sample", "switching_cycle", "head_pos_block",
  "head_pos_defs", "head_pos_defx", "head_pos_def", "head_pos_lowls",
  "head_pos_lowl", "headstack_pos", "if_block", "if_defs", "if_defx",
  "if_def", "if_lowls", "if_lowl", "if_def_st", "pass_order_block",
  "pass_order_defs", "pass_order_defx", "pass_order_def",
  "pass_order_lowls", "pass_order_lowl", "pass_order", "s2_group_order",
  "phase_cal_detect_block", "phase_cal_detect_defs",
  "phase_cal_detect_defx", "phase_cal_detect_def",
  "phase_cal_detect_lowls", "phase_cal_detect_lowl", "phase_cal_detect",
  "procedures_block", "procedures_defs", "procedures_defx",
  "procedures_def", "procedures_lowls", "procedures_lowl", "tape_change",
  "headstack_motion", "new_source_command", "new_tape_setup",
  "setup_always", "parity_check", "tape_prepass", "preob_cal", "midob_cal",
  "postob_cal", "procedure_name_prefix", "roll_block", "roll_defs",
  "roll_defx", "roll_def", "roll_lowls", "roll_lowl", "roll_reinit_period",
  "roll_inc_period", "roll", "roll_def_st", "scheduling_params_block",
  "scheduling_params_defs", "scheduling_params_defx",
  "scheduling_params_def", "scheduling_params_lowls",
  "scheduling_params_lowl", "sefd_block", "sefd_defs", "sefd_defx",
  "sefd_def", "sefd_lowls", "sefd_lowl", "sefd_model", "sefd",
  "site_block", "site_defs", "site_defx", "site_def", "site_lowls",
  "site_lowl", "site_type", "site_name", "site_id", "site_position",
  "site_position_epoch", "site_position_ref", "site_velocity",
  "horizon_map_az", "horizon_map_el", "zen_atmos", "ocean_load_vert",
  "ocean_load_horiz", "occupation_code", "inclination", "eccentricity",
  "arg_perigee", "ascending_node", "mean_anomaly", "semi_major_axis",
  "mean_motion", "orbit_epoch", "source_block", "source_defs",
  "source_defx", "source_def", "source_lowls", "source_lowl",
  "source_type", "source_name", "iau_name", "ra", "dec", "ref_coord_frame",
  "source_position_ref", "source_position_epoch", "ra_rate", "dec_rate",
  "velocity_wrt_lsr", "source_model", "tapelog_obs_block",
  "tapelog_obs_defs", "tapelog_obs_defx", "tapelog_obs_def",
  "tapelog_obs_lowls", "tapelog_obs_lowl", "vsn", "tracks_block",
  "tracks_defs", "tracks_defx", "tracks_def", "tracks_lowls",
  "tracks_lowl", "fanin_def", "fanout_def", "track_frame_format",
  "data_modulation", "vlba_frmtr_sys_trk", "vlba_trnsprt_sys_trk",
  "s2_recording_mode", "s2_data_source", "bit_stream_list", "external_ref",
  "literal", "unit_list", "unit_more", "unit_option", "unit_value",
  "name_list", "name_value", "value_list", "value", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,    61,    59,    58,    42
};
# endif

#define YYPACT_NINF -1360

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-1360)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      43,  -131, -1360, -1360,    29,    46, -1360, -1360,  -110, -1360,
    -114,  -106,  -103,  -101,   -93,   -59,   -54,   -52,   -47,   -45,
     -24,   -19,     8,    11,    13,    39,    42,    49,    52,    58,
      77,   108,   138, -1360,   788, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
     142,    48,    59,    65,    62,   161,   233,   317,   332,   349,
     396,   493,   633,   656,   662,   668,   680,   701,   711,   715,
     719,   754,   756,   758, -1360, -1360,   811, -1360, -1360,    48,
   -1360, -1360,    79, -1360, -1360,    59, -1360, -1360,   151, -1360,
   -1360,    65, -1360, -1360,   174, -1360, -1360,    62, -1360, -1360,
     176, -1360, -1360,   161, -1360, -1360,   188, -1360, -1360,   233,
   -1360, -1360,   192, -1360, -1360,   317, -1360, -1360,   196, -1360,
   -1360,   332, -1360, -1360,   207, -1360, -1360,   349, -1360, -1360,
     209, -1360, -1360,   396, -1360, -1360,   213, -1360, -1360,   493,
   -1360, -1360,   219, -1360, -1360,   633, -1360, -1360,   225, -1360,
   -1360,   656, -1360, -1360,   229, -1360, -1360,   662, -1360, -1360,
     235, -1360, -1360,   668, -1360, -1360,   239, -1360, -1360,   680,
   -1360, -1360,   243, -1360, -1360,   701, -1360, -1360,   248, -1360,
   -1360,   711, -1360, -1360,   254, -1360, -1360,   715, -1360, -1360,
     261, -1360, -1360,   719, -1360, -1360,   286, -1360, -1360,   754,
   -1360, -1360,   296, -1360, -1360,   756, -1360, -1360,   324, -1360,
   -1360,   758, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360,   156, -1360,   327, -1360,   338, -1360,
     340, -1360,   343, -1360,   345, -1360,   348, -1360,   354, -1360,
     363, -1360,   367, -1360,   388, -1360,   390, -1360,   392, -1360,
     414, -1360,   423, -1360,   427, -1360,   468, -1360,   474, -1360,
     476, -1360,   478, -1360,   501, -1360,   509, -1360,   513, -1360,
     524,   671,   746,   785,   224,    82,    50,   518,    56,   291,
      53,   352,   259,   164,   339,   315,   635,   565,   218,   159,
     319,   358,   619,   520,   551,   749,   811,   553, -1360, -1360,
     752, -1360, -1360,   555,   333,   562,   566,   583,   589, -1360,
   -1360,   822, -1360, -1360, -1360, -1360, -1360, -1360,   598,   595,
     599,   605,   607,   626,   636,   637,   640,   641,   648,   651,
     657,   665, -1360, -1360,   483, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,   621,
     634, -1360, -1360,   431, -1360, -1360, -1360,   655,   672,   674,
     676,   677,   685,   686,   687,   688,   690,   691,   692, -1360,
   -1360,   629, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360,   663,   693,   694,   695,   696,
     705,   706,   707,   708,   712,   716,   720,   721,   722,   723,
   -1360, -1360,   572, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,   670,
     724,   725,   726,   727, -1360, -1360,    73, -1360, -1360, -1360,
   -1360, -1360, -1360,   700,   728, -1360, -1360,   704, -1360, -1360,
   -1360,   710,   729,   730,   740,   770,   782,   805, -1360, -1360,
     660, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,   714,
     806, -1360, -1360,   733, -1360, -1360, -1360,   718,   807,   808,
     809,   810,   812,   813,   814,   815,   816, -1360, -1360,   622,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360,   732,   817, -1360, -1360,   736, -1360, -1360, -1360,   735,
     818,   819, -1360, -1360,   664, -1360, -1360, -1360, -1360,   738,
     823, -1360, -1360,   739, -1360, -1360, -1360,   742,   824,   825,
     826,   827, -1360, -1360,   654, -1360, -1360, -1360, -1360, -1360,
   -1360,   745,   828, -1360, -1360,   743, -1360, -1360, -1360,   748,
     829,   830, -1360, -1360,   228, -1360, -1360, -1360, -1360,   821,
     831,   832,   833,   834,   835,   836,   837,   838,   839,   840,
     841,   842,   843,   844,   845,   846,   847,   848,   849,   850,
     851, -1360, -1360,   364, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360,   852,   854,   855,
     856,   857,   858,   859,   860,   861,   862,   863,   864,   865,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
     491, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360,   866,   868,   869,   870,   871,
     872,   873,   874,   875, -1360, -1360,   569, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360,   876,   878, -1360,
   -1360,   625, -1360, -1360, -1360, -1360, -1360,   879,   881, -1360,
     882, -1360, -1360,   884,   887,   889,   890,   891,   886, -1360,
     227, -1360,  -110,   893,   894,   895,   896,   897,   898,   899,
     900,   901,   902,   904,   903, -1360, -1360, -1360,   905, -1360,
   -1360,   906,   906,   906,   907,   907,   907,   907,   907,   907,
     906,   908,   909, -1360, -1360,   906,   906,   910,  -110,   906,
    -124,   -58,   -57,   912,  -110,   906,   -21,   -20,   913,   911,
   -1360, -1360,   -18,   906,  -110,   914,   915, -1360, -1360,   -99,
     916, -1360, -1360,   906,   906,   906,   918,   880,   919,   917,
   -1360, -1360,   923,   920, -1360, -1360,   927,   928,  -110,  -110,
     929,   906,  -110,   930,   931,   926, -1360, -1360,  -110,   932,
   -1360, -1360,   907,  -110,   933, -1360, -1360,   934,   936, -1360,
   -1360,   906,  -110,   937,  -110,   938, -1360, -1360,   935,   939,
   -1360, -1360,   940,   942,   941, -1360, -1360,   946,   948,   949,
     906,   950,   951,   906,   906,   906,   906,   906,   906,   907,
     906,  -110,   906,   906,   906,   906,  -110,   952,   947, -1360,
   -1360,   954,   955,   956,   957,   958,   959,   906,   906,   961,
     962,   906,  -110,   960, -1360, -1360,   963,   -70,   964,   965,
    -110,  -110,   967,   968,   966, -1360, -1360,  -110,   969, -1360,
   -1360,   970, -1360,   971,   972,   973,   974, -1360,   975, -1360,
     811,   977,   978,   980,   981,   982,   983,   984,   985,   987,
     988,   989,   990, -1360, -1360,   976,   991,   992,   993, -1360,
     994,   995,   996,   997,   998,   999,  1001,  1002, -1360,  1003,
    1004,  1005,  1006,  1007, -1360,  1008,  1009, -1360,  1011, -1360,
    1012,  1013,  1014,  1015, -1360,  1016, -1360,  1017,  1018, -1360,
    1020,   906,  1019,  1022,  1023, -1360,  1024,   906, -1360,  1026,
    1027,  1028,  1029,  1030,  1031, -1360,  1032, -1360,  1034,  1035,
    1036,  1037,  1000,  -129,  1040,   -62,  1041, -1360,  1042, -1360,
     -15, -1360,   -13, -1360, -1360,    64, -1360,  1044,  1045,  1046,
      83, -1360,  1047, -1360,  1049,  1050, -1360,  1052,  1053,  1054,
    1056,  1055,  1058,  1059,  1061,  1062,  1063,  1064,  1065,  1068,
    1070,  1071,  1072,  1073,  1074,  1075,  1076,  1077, -1360,   110,
    1078,  1079,  1080,  1081,  1084,  1085,  1086,  1088,  1089,  1090,
    1091, -1360,  1092,  1093,   979,  1095,  1096,  1097,  1098,  1100,
     112, -1360,  1101, -1360,   114, -1360, -1360, -1360,   906,  1043,
     953, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360,   906,   906,   906,
     906,   906,   906, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360,  1048, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
     906,  1103, -1360, -1360,   906,   906,   147, -1360, -1360, -1360,
     906,  1107,  1111,  -110, -1360, -1360, -1360,   -61,  1106, -1360,
    1113, -1360, -1360,   906, -1360,   906, -1360,   907, -1360,  -110,
   -1360,  -110, -1360, -1360, -1360, -1360,  1114, -1360,   906, -1360,
   -1360, -1360,   906, -1360, -1360,   906, -1360, -1360, -1360,   906,
     906, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360,  1115, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360,  1069,  -110,   979,  1109,  1110, -1360, -1360,  1118,
    -110, -1360, -1360,  1102,  1119, -1360,  1120,   149,  1116, -1360,
    1117,  1121,  1122,  1123,  1124,  1125,  1126,  1127,   976,  1128,
   -1360, -1360, -1360,  1130,  1131,  1132,   178, -1360,  1139,  1133,
    1134,  1136,  1137,  1138,  -110, -1360,  1140,   210,  1142, -1360,
   -1360,   234,  1143,  1144,  1145,  1146,  1149,  1150,  1151,  1152,
    1153,  1154,  1158,   246,  1155,  1157,  1159,  1160, -1360, -1360,
    1164,   906,  1165,  1162, -1360, -1360, -1360,   907,   907,   907,
    1048,  1169,  1163, -1360, -1360,  1171,  1166,   906,   906, -1360,
    1148,  -110,  1168,  -110, -1360,   906, -1360, -1360,  1172,  -110,
     906,   906, -1360, -1360, -1360,   906,  -110,   246, -1360,  1170,
    1173,  -110, -1360,  1174,  1176, -1360,  1175, -1360,  1177, -1360,
    1179,  1180,  1181, -1360,  1182,   906,  1183,  1048,  1185,  1186,
    1188,  1189, -1360,  1190,  1191,  1192,   251,  1194,  1195,  1196,
    1197,  1198,  1178,  -110,   253,  1200,  1201,   906,  1202, -1360,
   -1360, -1360,   906,  1203,  1048,  1205, -1360,   906, -1360, -1360,
   -1360,   906,   906, -1360, -1360, -1360,   906,   979,  -110, -1360,
     282, -1360,  -110, -1360,  1207,  1206, -1360,  1209,  1208,  1211,
    1187,  1210, -1360,  1212,  1214,  1215,  1216,   285,   357, -1360,
    1218,  1219,  1221, -1360,  1202,  1213,  1220, -1360,  1226, -1360,
    1228,  -110, -1360,  1230, -1360, -1360, -1360, -1360,  1222,   373,
    1225,  1232,  1227,   395,  1229,  1234, -1360,  1237,  1238,  1233,
     906, -1360,  -109,   906, -1360,  1235, -1360,  1239,  1236,  1240,
    1241, -1360,  1243,   465,  1244,  -110, -1360,  1245,   608,   906,
   -1360, -1360,   -16,   906,  1246,  1247,   610, -1360,  -110,   612,
   -1360,  1249, -1360,  1250,  1251, -1360,  -110, -1360,   628, -1360,
   -1360, -1360, -1360,   925,   906, -1360, -1360,  1253,  1254, -1360,
   -1360
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       0,     0,     7,     8,     0,     3,     5,     6,     0,     1,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     2,    11,    12,    13,    14,    16,
      17,    18,    19,    20,    21,    22,    15,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,   675,
       0,    36,    38,    47,    91,   278,   475,   417,   231,   316,
     176,   134,   160,   195,   344,   382,   400,   453,   360,   490,
     508,   564,   627,   611,    10,     9,     0,    58,    59,    35,
      56,    57,     0,    42,    43,    37,    40,    41,     0,    51,
      52,    46,    49,    50,     0,    95,    96,    90,    93,    94,
       0,   282,   283,   277,   280,   281,     0,   479,   480,   474,
     477,   478,     0,   421,   422,   416,   419,   420,     0,   235,
     236,   230,   233,   234,     0,   320,   321,   315,   318,   319,
       0,   180,   181,   175,   178,   179,     0,   138,   139,   133,
     136,   137,     0,   164,   165,   159,   162,   163,     0,   199,
     200,   194,   197,   198,     0,   348,   349,   343,   346,   347,
       0,   386,   387,   381,   384,   385,     0,   404,   405,   399,
     402,   403,     0,   457,   458,   452,   455,   456,     0,   364,
     365,   359,   362,   363,     0,   494,   495,   489,   492,   493,
       0,   512,   513,   507,   510,   511,     0,   568,   569,   563,
     566,   567,     0,   631,   632,   626,   629,   630,     0,   615,
     616,   610,   613,   614,    61,    62,    63,    64,    65,    68,
      66,    67,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,     0,    55,     0,    39,     0,    48,
       0,    92,     0,   279,     0,   476,     0,   418,     0,   232,
       0,   317,     0,   177,     0,   135,     0,   161,     0,   196,
       0,   345,     0,   383,     0,   401,     0,   454,     0,   361,
       0,   491,     0,   509,     0,   565,     0,   628,     0,   612,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    84,    85,
       0,    82,    83,     0,     0,     0,     0,     0,     0,   106,
     107,     0,   100,   101,   102,   103,   104,   105,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   301,   302,     0,   287,   288,   289,   290,   291,
     292,   293,   294,   295,   296,   297,   298,   299,   300,     0,
       0,   487,   488,     0,   484,   485,   486,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   439,
     440,     0,   426,   427,   428,   429,   430,   431,   432,   433,
     434,   435,   436,   437,   438,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     256,   257,     0,   240,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,     0,
       0,     0,     0,     0,   331,   332,     0,   325,   326,   327,
     328,   329,   330,     0,     0,   188,   189,     0,   185,   186,
     187,     0,     0,     0,     0,     0,     0,     0,   151,   152,
       0,   143,   144,   145,   146,   147,   148,   149,   150,     0,
       0,   172,   173,     0,   169,   170,   171,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   215,   216,     0,
     204,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,     0,     0,   356,   357,     0,   353,   354,   355,     0,
       0,     0,   395,   396,     0,   391,   392,   393,   394,     0,
       0,   412,   413,     0,   409,   410,   411,     0,     0,     0,
       0,     0,   468,   469,     0,   462,   463,   464,   465,   466,
     467,     0,     0,   372,   373,     0,   369,   370,   371,     0,
       0,     0,   503,   504,     0,   499,   500,   501,   502,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   540,   541,     0,   517,   518,   519,   520,   521,   522,
     523,   524,   525,   526,   527,   528,   529,   530,   531,   532,
     533,   534,   535,   536,   537,   538,   539,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     595,   596,   586,   587,   588,   589,   590,   591,   592,   593,
       0,   573,   574,   575,   576,   577,   578,   579,   580,   581,
     582,   583,   584,   585,   594,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   646,   647,     0,   636,   637,   638,
     639,   640,   641,   642,   643,   644,   645,     0,     0,   623,
     624,     0,   620,   621,   622,    60,    45,     0,     0,    54,
       0,    81,    98,     0,     0,     0,     0,   123,     0,    99,
       0,   285,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   286,   482,   662,     0,   483,
     424,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   425,   238,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     239,   323,     0,     0,     0,     0,     0,   324,   183,     0,
       0,   184,   141,     0,     0,     0,     0,     0,     0,     0,
     142,   167,     0,     0,   168,   202,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   203,   351,     0,     0,
     352,   389,     0,     0,     0,   390,   407,     0,     0,   408,
     460,     0,     0,     0,     0,     0,   461,   367,     0,     0,
     368,   497,     0,     0,     0,   498,   515,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   516,
     571,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   572,   634,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   635,   618,     0,     0,   619,
      44,     0,    53,     0,     0,     0,     0,   124,     0,    97,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   284,   481,     0,     0,     0,     0,   672,
       0,     0,     0,     0,     0,     0,     0,     0,   423,     0,
       0,     0,     0,     0,   264,     0,   664,   266,     0,   268,
       0,     0,     0,     0,   273,     0,   275,     0,     0,   237,
       0,     0,     0,     0,     0,   322,     0,     0,   182,     0,
       0,     0,     0,     0,     0,   140,     0,   166,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   201,     0,   350,
       0,   671,     0,   674,   388,     0,   406,     0,     0,     0,
       0,   459,     0,   366,     0,     0,   496,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   514,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   570,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   633,     0,   617,     0,   108,   110,   109,     0,   125,
       0,   303,   304,   306,   307,   308,   309,   310,   311,   312,
     313,   314,   305,   669,   441,   443,   444,     0,     0,     0,
       0,     0,     0,   442,   451,   258,   259,   260,   261,   262,
     263,     0,   265,   267,   269,   270,   271,   272,   274,   276,
       0,     0,   340,   341,     0,     0,     0,   153,   154,   156,
       0,     0,     0,     0,   217,   218,   219,     0,     0,   223,
       0,   225,   226,     0,   229,     0,   397,     0,   398,     0,
     415,     0,   470,   471,   472,   473,     0,   505,     0,   542,
     543,   544,     0,   546,   547,     0,   549,   550,   551,     0,
       0,   554,   555,   556,   557,   558,   559,   560,   561,   562,
     597,     0,   599,   600,   601,   602,   604,   606,   607,   605,
     603,   608,     0,     0,     0,     0,     0,   651,   652,     0,
       0,   656,   658,     0,     0,    87,     0,     0,     0,   126,
       0,     0,     0,     0,     0,     0,     0,     0,   675,   663,
     666,   667,   668,     0,     0,     0,     0,   190,     0,     0,
       0,     0,     0,     0,     0,   222,     0,     0,     0,   670,
     673,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    89,    86,
       0,     0,   127,     0,   445,   446,   447,     0,     0,     0,
       0,     0,     0,   342,   191,     0,     0,     0,     0,   155,
       0,     0,     0,     0,   227,     0,   358,   414,     0,     0,
       0,     0,   552,   553,   598,     0,     0,     0,   660,     0,
       0,     0,   655,     0,     0,    88,     0,   128,     0,   661,
       0,     0,     0,   665,     0,     0,     0,     0,     0,     0,
       0,     0,   221,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   114,   129,   448,
     449,   450,     0,     0,     0,     0,   157,     0,   174,   220,
     224,     0,     0,   506,   545,   548,     0,     0,     0,   659,
       0,   654,     0,   657,     0,     0,   115,     0,     0,     0,
       0,     0,   193,     0,     0,     0,     0,     0,     0,   650,
       0,     0,   116,   130,   129,     0,     0,   192,     0,   228,
       0,     0,   648,     0,   649,   653,   625,   117,     0,     0,
       0,     0,     0,     0,     0,   118,   113,   131,     0,     0,
       0,   375,     0,     0,   119,     0,   132,     0,     0,     0,
       0,   377,     0,     0,     0,   120,   112,     0,     0,     0,
     376,   378,     0,     0,     0,   121,     0,   335,     0,     0,
     338,     0,   379,     0,     0,   111,     0,   333,     0,   339,
     336,   337,   158,   374,     0,   122,   334,     0,     0,   380,
     609
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
   -1360, -1360, -1360,  1242, -1360, -1360,  1021, -1360, -1360, -1360,
    1108, -1360, -1360, -1360,  1105, -1360,   675,   -78, -1360,  -312,
   -1360,   750, -1360, -1360, -1360, -1360,  1224, -1360, -1360,   753,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360,  -327, -1360, -1360, -1360,  1066, -1360, -1360,   596,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,  1010, -1360,
   -1360,   538, -1360, -1360, -1360,  1129, -1360, -1360,   666, -1360,
   -1360, -1360,  1135, -1360, -1360,   620, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360,  1231, -1360, -1360,
     699, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360,  1281, -1360, -1360,
     921, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360,  1156, -1360, -1360,   678, -1360,
    -207, -1359, -1360, -1360, -1360, -1360, -1360,  1248, -1360, -1360,
     761, -1360, -1360, -1360,  1223, -1360, -1360,   630, -1360, -1360,
   -1360,  1104, -1360, -1360,   883, -1360, -1360, -1360, -1360,  1252,
   -1360, -1360,   796, -1360, -1360, -1360,  1217, -1360, -1360,  1025,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360,  1255, -1360, -1360,   803, -1360, -1360, -1360,
   -1360, -1360, -1360,  1278, -1360, -1360,  1038, -1360, -1360,  1256,
   -1360, -1360,   853, -1360, -1360, -1360, -1360,  1257, -1360, -1360,
     820, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360, -1360, -1360,  -304,  -301,  -296,  -293,  -292,  -289,
    -288,  -287, -1360, -1360,  1258, -1360, -1360,   760, -1360, -1360,
   -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1360,  1199, -1360, -1360,   731, -1360, -1360, -1360,  1204,
   -1360, -1360,   755, -1360, -1360, -1360, -1360, -1360, -1360, -1360,
   -1360, -1124,  -150, -1360,  -734, -1360, -1189,  -720, -1360,  -691,
    -799,    -8
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     4,     5,     6,     7,    34,    35,    36,    37,    95,
      96,    97,    38,   101,   102,   103,    89,    90,    91,   244,
     320,   321,   322,  1157,    39,   107,   108,   109,   331,   332,
     333,   334,   335,   336,   337,  1315,  1348,  1365,  1384,   878,
    1160,  1258,  1318,  1367,    40,   149,   150,   151,   470,   471,
     472,   473,   474,   475,   476,   477,    41,   155,   156,   157,
     483,   484,   485,    42,   143,   144,   145,   457,   458,   459,
      43,   161,   162,   163,   499,   500,   501,   502,   503,   504,
     505,   506,   507,   508,   509,    44,   131,   132,   133,   422,
     423,   424,   425,   426,   427,   428,   429,   430,   431,   432,
     433,   434,   435,   436,   437,    45,   113,   114,   115,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,    46,   137,   138,   139,   446,   447,   448,
    1389,  1390,   449,   450,   451,    47,   167,   168,   169,   515,
     516,   517,    48,   191,   192,   193,   555,   556,   557,    49,
     173,   174,   175,   524,   525,   526,   527,    50,   179,   180,
     181,   533,   534,   535,    51,   125,   126,   127,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,    52,   185,   186,   187,   544,   545,   546,   547,   548,
     549,    53,   119,   120,   121,   373,   374,    54,   197,   198,
     199,   564,   565,   566,   567,    55,   203,   204,   205,   593,
     594,   595,   596,   597,   598,   599,   600,   601,   602,   603,
     604,   605,   606,   607,   608,   609,   610,   611,   612,   613,
     614,   615,    56,   209,   210,   211,   640,   641,   642,   643,
     644,   645,   646,   647,   648,   649,   650,   651,   652,   653,
      57,   221,   222,   223,   681,   682,   683,    58,   215,   216,
     217,   666,   667,   668,   669,   670,   671,   672,   673,   674,
     675,  1146,   368,   376,   915,  1169,  1170,   916,   960,   900,
     962,   963
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint16 yytable[] =
{
      60,   896,   897,   898,   688,   970,   632,   918,   920,   633,
     906,   245,   925,   927,   634,   909,   910,   635,   636,   913,
    1201,   637,   638,   639,     8,   923,   895,  1089,  1090,     9,
    1401,  1263,   914,   932,   901,   902,   903,   904,   905,  1401,
      59,   895,    61,   939,   940,   941,     1,  1371,  1372,     1,
      62,   936,    86,    63,   338,    64,   377,   338,   937,   461,
     338,   953,   439,    65,    92,   440,   441,   442,   443,   104,
      98,   462,   463,   464,   465,   466,   467,   338,  1295,   756,
    1013,   967,   440,   441,   442,   443,   338,  1014,   369,  1183,
     984,   985,   895,   895,  1092,  1093,  1184,    66,   917,   919,
     980,   961,    67,   983,    68,  1321,   986,   987,   988,    69,
     990,    70,   992,   993,   994,   995,   378,   379,   380,   381,
     382,   383,   384,   385,   386,   387,   388,  1005,  1006,   895,
     895,  1009,    71,   930,   895,   924,   926,    72,   989,   931,
    1392,  1096,  1097,  1098,  1099,   375,   404,   438,   452,   460,
     478,   486,   510,   518,   528,   536,   550,   558,   568,   616,
     654,   676,   684,   338,    73,   569,   110,    74,   338,    75,
     511,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,  1327,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    76,     2,     3,    77,     2,
       3,    87,    88,   389,   390,    78,   468,   469,    79,   444,
     445,  1071,    93,    94,    80,   105,   106,  1076,    99,   100,
    1100,  1101,   338,   375,   559,   512,   444,   445,   338,   246,
     339,   370,   338,    81,   814,   371,   372,   245,   116,  1105,
    1099,   404,   570,   571,   572,   573,   574,   575,   576,   577,
     578,   579,   580,   581,   582,   583,   584,   585,   586,   587,
     588,   589,   590,   338,    82,   487,  1130,  1131,  1152,  1153,
    1155,  1156,   438,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,   350,   351,   488,   489,   490,   491,   492,
     493,   494,   495,   496,    83,   338,   452,   453,    85,   560,
     561,   248,  1191,  1177,  1178,  1209,  1210,   460,  1158,   560,
     561,   290,   591,   592,   111,   112,   454,   513,   514,   338,
     478,   529,   122,   338,   250,   617,   252,  1162,  1163,  1164,
    1165,  1166,  1167,   486,  1224,  1225,   632,   128,   254,   633,
    1175,  1171,   256,   338,   634,   519,   258,   635,   636,   510,
    1173,   637,   638,   639,   134,  1176,   338,   260,   479,   262,
    1179,  1188,   338,   264,   655,   518,  1234,  1235,   338,   266,
     838,   562,   563,  1187,   528,   268,   480,   352,   353,   270,
     530,   562,   563,   536,   880,   272,   117,   118,  1193,   274,
    1237,  1099,  1194,   276,   550,  1195,    59,  1249,   278,  1196,
    1197,   140,   520,   521,   280,   558,  1189,  1303,  1099,  1311,
    1312,   282,   497,   498,   568,   583,   584,   585,   586,   587,
     588,   589,   590,   618,   619,   620,   621,   622,   623,   624,
     625,   626,   627,   628,   629,   338,   284,   718,  1329,  1099,
    1276,  1342,  1343,   616,   455,   456,   286,   570,   571,   572,
     573,   574,   575,   576,   577,   578,   579,   580,   581,   582,
     583,   584,   585,   586,   587,   588,   589,   590,   531,   532,
     123,   124,   630,   631,   288,   656,   657,   658,   659,   660,
     661,   662,   663,   291,  1310,   129,   130,   338,   693,   714,
     654,  1256,   522,   523,   292,   338,   293,   853,   146,   294,
    1171,   295,   135,   136,   296,   481,   482,  1268,  1269,  1328,
     297,   664,   665,  1344,  1099,  1274,   676,   591,   592,   298,
    1277,  1278,   338,   299,   405,  1279,  1260,  1261,  1262,  1356,
    1357,   684,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   350,   351,   300,  1293,   301,  1171,   302,   141,
     142,  1361,  1362,   406,   407,   408,   409,   410,   411,   412,
     413,   414,   415,   416,   417,   418,   419,  1316,  1030,   338,
     303,   551,  1319,   338,  1171,   864,   338,  1323,   749,   304,
     370,  1324,  1325,   305,   371,   372,  1326,   583,   584,   585,
     586,   587,   588,   589,   590,   618,   619,   620,   621,   622,
     623,   624,   625,   626,   627,   628,   629,   406,   407,   408,
     409,   410,   411,   412,   413,   414,   415,   416,   417,   418,
     419,  1381,  1382,   338,   306,   677,   338,   552,   785,   338,
     307,   868,   308,   338,   309,   732,   352,   353,   152,   338,
    1370,   537,  1373,  1374,   630,   631,   147,   148,   488,   489,
     490,   491,   492,   493,   494,   495,   496,   310,   338,  1391,
     805,   158,  1393,  1394,   338,   311,   769,   164,   338,   312,
     794,   420,   421,   170,   313,    86,   685,   314,   462,   463,
     464,   465,   466,   467,  1408,   176,   656,   657,   658,   659,
     660,   661,   662,   663,   881,   378,   379,   380,   381,   382,
     383,   384,   385,   386,   387,   388,   182,   686,   338,   689,
     760,   692,   538,   539,   540,   541,   188,   694,   553,   554,
     194,   695,   664,   665,   200,   420,   421,   520,   521,   454,
     912,   538,   539,   540,   541,   678,   922,   338,   696,   773,
     338,   678,   789,   338,   697,   798,   933,   338,   700,   809,
     316,   701,   317,    86,   702,   687,   316,   480,   690,   206,
     703,   212,   704,   218,  1387,  1388,  1397,  1388,  1400,  1388,
     950,   951,   679,   680,   954,   497,   498,   716,   679,   680,
     958,   705,   389,   390,  1406,  1388,   153,   154,   542,   543,
     717,   706,   707,   323,   968,   708,   709,   512,   324,   325,
     326,   327,   328,   710,   530,   552,   711,   542,   543,   159,
     160,   720,   712,   468,   469,   165,   166,   522,   523,   734,
     713,   171,   172,   991,    87,    88,   751,   721,   996,   722,
     698,   723,   724,   177,   178,   324,   325,   326,   327,   328,
     725,   726,   727,   728,  1010,   729,   730,   731,   735,   736,
     737,   738,  1017,  1018,   183,   184,   758,   455,   456,  1022,
     739,   740,   741,   742,   189,   190,   762,   743,   195,   196,
     771,   744,   201,   202,   775,   745,   746,   747,   748,   752,
     753,   754,   755,   759,   763,   764,   481,   482,   787,   513,
     514,   791,   531,   532,   796,   765,   553,   554,   800,   318,
     319,   807,    87,    88,   811,   318,   319,   207,   208,   213,
     214,   219,   220,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,   766,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,   767,   329,   330,
     224,   225,   226,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     768,   772,   776,   777,   778,   779,   315,   780,   781,   782,
     783,   784,   788,   792,   793,   329,   330,   816,   797,   801,
     802,   803,   804,   808,   812,   813,   817,   818,   819,   820,
     821,   822,   823,   824,   825,   826,   827,   828,   829,   830,
     831,   832,   833,   834,   835,   836,   837,  1349,   840,   841,
     842,   843,   844,   845,   846,   847,   848,   849,   850,   851,
     852,   774,   855,   856,   857,   858,   859,   860,   861,   862,
     863,   943,   866,   867,   873,   870,   871,   874,   872,   875,
     876,   877,   879,   882,   883,   884,   885,   886,   887,   888,
     889,   890,   891,  1172,   892,    84,   895,   899,   907,   893,
     911,   894,   921,   928,   934,   908,   770,   929,   942,   944,
     691,   935,   938,   945,   946,  1182,   947,   948,   949,   952,
     955,   956,   957,  1407,   699,   965,   972,   969,   959,   964,
     974,  1190,   966,   975,   971,   973,   977,   976,   978,   979,
     981,   982,   997,   998,   999,  1000,  1001,  1002,  1161,  1004,
    1003,  1007,  1008,  1012,  1015,  1016,  1011,  1019,  1020,   786,
    1024,   750,  1021,   761,   757,  1023,  1043,  1025,  1026,  1027,
    1145,  1028,  1029,  1031,  1032,  1200,  1033,  1034,  1035,  1036,
    1037,  1038,  1205,  1039,  1040,  1041,  1042,  1044,  1045,  1046,
    1088,  1047,  1048,  1049,  1050,  1051,  1052,  1053,  1054,  1055,
    1056,  1057,  1058,  1059,  1060,   267,  1061,  1062,  1063,  1064,
    1065,  1066,  1067,  1068,  1069,  1072,  1232,  1070,  1073,  1398,
    1074,  1075,  1077,  1078,  1079,   810,  1080,  1081,  1082,  1083,
    1084,  1085,  1086,  1159,  1087,  1250,  1091,  1094,  1168,  1095,
    1102,  1103,  1104,   247,  1106,  1107,   249,  1108,  1109,  1110,
    1111,  1113,  1172,  1112,  1114,   265,  1115,  1116,  1117,  1118,
    1199,  1119,  1120,  1271,  1121,  1273,  1122,  1123,  1124,  1125,
    1126,  1127,  1128,  1129,  1132,  1133,  1134,  1135,  1280,  1281,
    1136,  1137,  1138,  1284,  1139,  1140,  1141,    33,  1142,  1143,
    1144,  1147,  1148,  1206,  1149,  1150,  1151,  1180,  1154,  1172,
    1174,  1181,  1185,  1186,  1192,  1198,  1202,  1203,  1204,  1207,
    1208,  1213,   263,  1211,  1212,   715,   790,   273,  1214,  1215,
    1216,  1222,  1217,  1218,  1219,  1220,  1172,  1221,  1223,  1226,
    1227,  1228,  1229,   261,  1230,  1231,   269,  1233,  1236,  1270,
    1238,  1239,  1240,  1241,  1330,  1242,  1243,  1244,  1248,  1245,
    1246,  1247,  1251,  1252,  1255,  1257,  1253,  1254,  1259,  1264,
    1265,  1266,  1275,  1267,  1272,  1285,  1286,  1282,  1309,   799,
    1283,   251,  1287,  1354,  1288,  1289,  1290,  1291,  1336,  1292,
    1294,  1296,   257,  1297,  1298,  1299,  1300,   806,  1301,  1302,
    1304,  1305,  1317,  1306,  1307,  1308,  1313,  1331,  1314,  1333,
    1320,  1322,   259,  1332,  1350,  1334,  1337,  1385,  1335,  1338,
    1339,  1347,  1340,  1341,  1345,  1346,  1352,  1351,  1353,  1355,
    1399,  1249,  1358,  1359,  1360,  1364,  1363,  1366,  1405,  1368,
    1369,  1378,  1375,  1377,   253,  1376,  1386,   255,  1379,  1380,
     854,  1383,  1395,  1409,  1396,  1402,  1403,   795,  1404,     0,
    1410,   719,   869,   839,   279,   271,   733,   815,     0,   287,
     289,   865,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   275,     0,     0,     0,     0,     0,     0,     0,     0,
     277,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   281,     0,     0,     0,     0,     0,     0,
     283,     0,     0,     0,     0,     0,     0,   285
};

static const yytype_int16 yycheck[] =
{
       8,   721,   722,   723,   316,   804,   310,   741,   742,   310,
     730,    89,   746,   747,   310,   735,   736,   310,   310,   739,
    1144,   310,   310,   310,   155,   745,   150,   156,   157,     0,
    1389,  1220,   156,   753,   725,   726,   727,   728,   729,  1398,
     150,   150,   156,   763,   764,   765,     3,   156,   157,     3,
     156,   150,     4,   156,     4,   156,     6,     4,   157,     6,
       4,   781,     6,   156,     5,     9,    10,    11,    12,     7,
       5,    18,    19,    20,    21,    22,    23,     4,  1267,     6,
     150,   801,     9,    10,    11,    12,     4,   157,     6,   150,
     824,   825,   150,   150,   156,   157,   157,   156,   156,   156,
     820,   792,   156,   823,   156,  1294,   826,   827,   828,   156,
     830,   156,   832,   833,   834,   835,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,   847,   848,   150,
     150,   851,   156,   151,   150,   156,   156,   156,   829,   157,
     156,   156,   157,   156,   157,   295,   296,   297,   298,   299,
     300,   301,   302,   303,   304,   305,   306,   307,   308,   309,
     310,   311,   312,     4,   156,     6,     5,   156,     4,   156,
       6,   125,   126,   127,   128,   129,   130,   131,   132,   133,
     134,   135,   136,  1307,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   156,   153,   154,   156,   153,
     154,   153,   154,   153,   154,   156,   153,   154,   156,   153,
     154,   931,   153,   154,   156,   153,   154,   937,   153,   154,
     156,   157,     4,   373,     6,    61,   153,   154,     4,   150,
       6,   149,     4,   156,     6,   153,   154,   315,     5,   156,
     157,   391,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,     4,   156,     6,   156,   157,   156,   157,
     156,   157,   422,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    26,    27,    28,    29,    30,
      31,    32,    33,    34,   156,     4,   446,     6,   156,    81,
      82,   150,  1101,   156,   157,   156,   157,   457,  1028,    81,
      82,   155,   153,   154,   153,   154,    25,   153,   154,     4,
     470,     6,     5,     4,   150,     6,   150,  1047,  1048,  1049,
    1050,  1051,  1052,   483,   156,   157,   640,     5,   150,   640,
    1074,  1061,   150,     4,   640,     6,   150,   640,   640,   499,
    1070,   640,   640,   640,     5,  1075,     4,   150,     6,   150,
    1080,  1095,     4,   150,     6,   515,   156,   157,     4,   150,
       6,   153,   154,  1093,   524,   150,    24,   153,   154,   150,
      65,   153,   154,   533,   157,   150,   153,   154,  1108,   150,
     156,   157,  1112,   150,   544,  1115,   150,   151,   150,  1119,
    1120,     5,    63,    64,   150,   555,  1097,   156,   157,   156,
     157,   150,   153,   154,   564,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,     4,   150,     6,   156,   157,
    1239,   156,   157,   593,   153,   154,   150,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   153,   154,
     153,   154,   153,   154,   150,   117,   118,   119,   120,   121,
     122,   123,   124,   156,  1283,   153,   154,     4,   155,     6,
     640,  1211,   153,   154,   156,     4,   156,     6,     5,   156,
    1220,   156,   153,   154,   156,   153,   154,  1227,  1228,  1308,
     156,   153,   154,   156,   157,  1235,   666,   153,   154,   156,
    1240,  1241,     4,   156,     6,  1245,  1217,  1218,  1219,   156,
     157,   681,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,   156,  1265,   156,  1267,   156,   153,
     154,   156,   157,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,  1287,   880,     4,
     156,     6,  1292,     4,  1294,     6,     4,  1297,     6,   156,
     149,  1301,  1302,   156,   153,   154,  1306,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,   156,   157,     4,   156,     6,     4,    62,     6,     4,
     156,     6,   156,     4,   156,     6,   153,   154,     5,     4,
    1360,     6,  1362,  1363,   153,   154,   153,   154,    26,    27,
      28,    29,    30,    31,    32,    33,    34,   156,     4,  1379,
       6,     5,  1382,  1383,     4,   156,     6,     5,     4,   156,
       6,   153,   154,     5,   150,     4,   156,     6,    18,    19,
      20,    21,    22,    23,  1404,     5,   117,   118,   119,   120,
     121,   122,   123,   124,   702,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,     5,   156,     4,   156,
       6,   156,    77,    78,    79,    80,     5,   155,   153,   154,
       5,   155,   153,   154,     5,   153,   154,    63,    64,    25,
     738,    77,    78,    79,    80,   116,   744,     4,   155,     6,
       4,   116,     6,     4,   155,     6,   754,     4,   150,     6,
       4,   156,     6,     4,   155,     6,     4,    24,     6,     5,
     155,     5,   155,     5,   156,   157,   156,   157,   156,   157,
     778,   779,   153,   154,   782,   153,   154,   156,   153,   154,
     788,   155,   153,   154,   156,   157,   153,   154,   153,   154,
     156,   155,   155,     8,   802,   155,   155,    61,    13,    14,
      15,    16,    17,   155,    65,    62,   155,   153,   154,   153,
     154,   156,   155,   153,   154,   153,   154,   153,   154,   156,
     155,   153,   154,   831,   153,   154,   156,   155,   836,   155,
       8,   155,   155,   153,   154,    13,    14,    15,    16,    17,
     155,   155,   155,   155,   852,   155,   155,   155,   155,   155,
     155,   155,   860,   861,   153,   154,   156,   153,   154,   867,
     155,   155,   155,   155,   153,   154,   156,   155,   153,   154,
     156,   155,   153,   154,   156,   155,   155,   155,   155,   155,
     155,   155,   155,   155,   155,   155,   153,   154,   156,   153,
     154,   156,   153,   154,   156,   155,   153,   154,   156,   153,
     154,   156,   153,   154,   156,   153,   154,   153,   154,   153,
     154,   153,   154,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   155,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   155,   153,   154,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     155,   155,   155,   155,   155,   155,   291,   155,   155,   155,
     155,   155,   155,   155,   155,   153,   154,   156,   155,   155,
     155,   155,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   155,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   155,   155,   155,   155,   155,   155,  1334,   156,   155,
     155,   155,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   483,   156,   155,   155,   155,   155,   155,   155,   155,
     155,   151,   156,   155,   150,   156,   155,   150,   156,   150,
     150,   150,   156,   150,   150,   150,   150,   150,   150,   150,
     150,   150,   150,  1061,   150,    34,   150,   150,   150,   156,
     150,   156,   150,   150,   150,   156,   470,   156,   150,   150,
     320,   156,   156,   156,   151,  1083,   156,   150,   150,   150,
     150,   150,   156,   158,   331,   151,   151,   150,   156,   156,
     150,  1099,   156,   151,   156,   156,   150,   156,   150,   150,
     150,   150,   150,   156,   150,   150,   150,   150,   155,   150,
     152,   150,   150,   150,   150,   150,   156,   150,   150,   499,
     150,   422,   156,   457,   446,   156,   150,   156,   156,   156,
     151,   157,   157,   156,   156,  1143,   156,   156,   156,   156,
     156,   156,  1150,   156,   156,   156,   156,   156,   156,   156,
     150,   157,   157,   157,   157,   157,   157,   156,   156,   156,
     156,   156,   156,   156,   156,   155,   157,   156,   156,   156,
     156,   156,   156,   156,   156,   156,  1184,   157,   156,  1386,
     157,   157,   156,   156,   156,   555,   157,   157,   157,   157,
     156,   156,   156,   150,   157,  1203,   156,   156,   150,   157,
     156,   156,   156,    95,   157,   156,   101,   157,   156,   156,
     156,   156,  1220,   157,   156,   149,   157,   156,   156,   156,
     151,   157,   157,  1231,   156,  1233,   156,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   156,   156,  1246,  1247,
     156,   156,   156,  1251,   156,   156,   156,     5,   157,   157,
     157,   156,   156,   151,   157,   157,   156,   150,   157,  1267,
     157,   150,   156,   150,   150,   150,   157,   157,   150,   150,
     150,   150,   143,   157,   157,   354,   515,   173,   156,   156,
     156,   150,   157,   157,   157,   157,  1294,   157,   156,   150,
     157,   157,   156,   137,   157,   157,   161,   157,   156,   151,
     157,   157,   157,   157,  1312,   156,   156,   156,   150,   157,
     157,   157,   157,   156,   150,   150,   157,   157,   156,   150,
     157,   150,   150,   157,   156,   151,   150,   157,   150,   533,
     157,   107,   157,  1341,   157,   156,   156,   156,   151,   157,
     157,   156,   125,   157,   156,   156,   156,   544,   157,   157,
     156,   156,   150,   157,   157,   157,   156,   150,   157,   150,
     157,   156,   131,   157,   151,   157,   156,  1375,   157,   157,
     156,   150,   157,   157,   156,   156,   150,   157,   150,   157,
    1388,   151,   157,   151,   157,   151,   157,   150,  1396,   151,
     157,   151,   157,   157,   113,   156,   151,   119,   157,   156,
     640,   157,   156,   150,   157,   156,   156,   524,   157,    -1,
     156,   373,   681,   593,   191,   167,   391,   564,    -1,   215,
     221,   666,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   179,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     185,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   197,    -1,    -1,    -1,    -1,    -1,    -1,
     203,    -1,    -1,    -1,    -1,    -1,    -1,   209
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,     3,   153,   154,   160,   161,   162,   163,   155,     0,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   162,   164,   165,   166,   167,   171,   183,
     203,   215,   222,   229,   244,   264,   282,   294,   301,   308,
     316,   323,   340,   350,   356,   364,   391,   409,   416,   150,
     440,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   165,   156,     4,   153,   154,   175,
     176,   177,     5,   153,   154,   168,   169,   170,     5,   153,
     154,   172,   173,   174,     7,   153,   154,   184,   185,   186,
       5,   153,   154,   265,   266,   267,     5,   153,   154,   351,
     352,   353,     5,   153,   154,   324,   325,   326,     5,   153,
     154,   245,   246,   247,     5,   153,   154,   283,   284,   285,
       5,   153,   154,   223,   224,   225,     5,   153,   154,   204,
     205,   206,     5,   153,   154,   216,   217,   218,     5,   153,
     154,   230,   231,   232,     5,   153,   154,   295,   296,   297,
       5,   153,   154,   309,   310,   311,     5,   153,   154,   317,
     318,   319,     5,   153,   154,   341,   342,   343,     5,   153,
     154,   302,   303,   304,     5,   153,   154,   357,   358,   359,
       5,   153,   154,   365,   366,   367,     5,   153,   154,   392,
     393,   394,     5,   153,   154,   417,   418,   419,     5,   153,
     154,   410,   411,   412,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   178,   176,   150,   169,   150,   173,
     150,   185,   150,   266,   150,   352,   150,   325,   150,   246,
     150,   284,   150,   224,   150,   205,   150,   217,   150,   231,
     150,   296,   150,   310,   150,   318,   150,   342,   150,   303,
     150,   358,   150,   366,   150,   393,   150,   418,   150,   411,
     155,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   150,     6,   175,     4,     6,   153,   154,
     179,   180,   181,     8,    13,    14,    15,    16,    17,   153,
     154,   187,   188,   189,   190,   191,   192,   193,     4,     6,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,   153,   154,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   431,     6,
     149,   153,   154,   354,   355,   431,   432,     6,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,   153,
     154,   327,   328,   329,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   431,     6,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
     153,   154,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   431,     6,
       9,    10,    11,    12,   153,   154,   286,   287,   288,   291,
     292,   293,   431,     6,    25,   153,   154,   226,   227,   228,
     431,     6,    18,    19,    20,    21,    22,    23,   153,   154,
     207,   208,   209,   210,   211,   212,   213,   214,   431,     6,
      24,   153,   154,   219,   220,   221,   431,     6,    26,    27,
      28,    29,    30,    31,    32,    33,    34,   153,   154,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     431,     6,    61,   153,   154,   298,   299,   300,   431,     6,
      63,    64,   153,   154,   312,   313,   314,   315,   431,     6,
      65,   153,   154,   320,   321,   322,   431,     6,    77,    78,
      79,    80,   153,   154,   344,   345,   346,   347,   348,   349,
     431,     6,    62,   153,   154,   305,   306,   307,   431,     6,
      81,    82,   153,   154,   360,   361,   362,   363,   431,     6,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   153,   154,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   431,     6,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     153,   154,   383,   384,   385,   386,   387,   388,   389,   390,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   431,     6,   117,   118,   119,   120,
     121,   122,   123,   124,   153,   154,   420,   421,   422,   423,
     424,   425,   426,   427,   428,   429,   431,     6,   116,   153,
     154,   413,   414,   415,   431,   156,   156,     6,   178,   156,
       6,   180,   156,   155,   155,   155,   155,   155,     8,   188,
     150,   156,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   155,   155,   155,     6,   269,   156,   156,     6,   355,
     156,   155,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   155,     6,   328,   156,   155,   155,   155,   155,   155,
     155,   155,   155,   155,   155,   155,   155,   155,   155,     6,
     249,   156,   155,   155,   155,   155,     6,   287,   156,   155,
       6,   227,   156,   155,   155,   155,   155,   155,   155,     6,
     208,   156,   155,     6,   220,   156,   155,   155,   155,   155,
     155,   155,   155,   155,   155,     6,   234,   156,   155,     6,
     299,   156,   155,   155,     6,   313,   156,   155,     6,   321,
     156,   155,   155,   155,   155,     6,   345,   156,   155,     6,
     306,   156,   155,   155,     6,   361,   156,   155,   155,   155,
     155,   155,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   155,   155,   155,   155,   155,   155,   155,     6,   369,
     156,   155,   155,   155,   155,   155,   155,   155,   155,   155,
     155,   155,   155,     6,   396,   156,   155,   155,   155,   155,
     155,   155,   155,   155,     6,   421,   156,   155,     6,   414,
     156,   155,   156,   150,   150,   150,   150,   150,   198,   156,
     157,   440,   150,   150,   150,   150,   150,   150,   150,   150,
     150,   150,   150,   156,   156,   150,   436,   436,   436,   150,
     438,   438,   438,   438,   438,   438,   436,   150,   156,   436,
     436,   150,   440,   436,   156,   433,   436,   156,   433,   156,
     433,   150,   440,   436,   156,   433,   156,   433,   150,   156,
     151,   157,   436,   440,   150,   156,   150,   157,   156,   436,
     436,   436,   150,   151,   150,   156,   151,   156,   150,   150,
     440,   440,   150,   436,   440,   150,   150,   156,   440,   156,
     437,   438,   439,   440,   156,   151,   156,   436,   440,   150,
     439,   156,   151,   156,   150,   151,   156,   150,   150,   150,
     436,   150,   150,   436,   433,   433,   436,   436,   436,   438,
     436,   440,   436,   436,   436,   436,   440,   150,   156,   150,
     150,   150,   150,   152,   150,   436,   436,   150,   150,   436,
     440,   156,   150,   150,   157,   150,   150,   440,   440,   150,
     150,   156,   440,   156,   150,   156,   156,   156,   157,   157,
     178,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   150,   156,   156,   156,   157,   157,   157,
     157,   157,   157,   156,   156,   156,   156,   156,   156,   156,
     156,   157,   156,   156,   156,   156,   156,   156,   156,   156,
     157,   436,   156,   156,   157,   157,   436,   156,   156,   156,
     157,   157,   157,   157,   156,   156,   156,   157,   150,   156,
     157,   156,   156,   157,   156,   157,   156,   157,   156,   157,
     156,   157,   156,   156,   156,   156,   157,   156,   157,   156,
     156,   156,   157,   156,   156,   157,   156,   156,   156,   157,
     157,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   157,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   157,   157,   157,   151,   430,   156,   156,   157,
     157,   156,   156,   157,   157,   156,   157,   182,   436,   150,
     199,   155,   436,   436,   436,   436,   436,   436,   150,   434,
     435,   436,   440,   436,   157,   433,   436,   156,   157,   436,
     150,   150,   440,   150,   157,   156,   150,   436,   433,   438,
     440,   439,   150,   436,   436,   436,   436,   436,   150,   151,
     440,   430,   157,   157,   150,   440,   151,   150,   150,   156,
     157,   157,   157,   150,   156,   156,   156,   157,   157,   157,
     157,   157,   150,   156,   156,   157,   150,   157,   157,   156,
     157,   157,   440,   157,   156,   157,   156,   156,   157,   157,
     157,   157,   156,   156,   156,   157,   157,   157,   150,   151,
     440,   157,   156,   157,   157,   150,   436,   150,   200,   156,
     438,   438,   438,   435,   150,   157,   150,   157,   436,   436,
     151,   440,   156,   440,   436,   150,   439,   436,   436,   436,
     440,   440,   157,   157,   440,   151,   150,   157,   157,   156,
     156,   156,   157,   436,   157,   435,   156,   157,   156,   156,
     156,   157,   157,   156,   156,   156,   157,   157,   157,   150,
     439,   156,   157,   156,   157,   194,   436,   150,   201,   436,
     157,   435,   156,   436,   436,   436,   436,   430,   439,   156,
     440,   150,   157,   150,   157,   157,   151,   156,   157,   156,
     157,   157,   156,   157,   156,   156,   156,   150,   195,   201,
     151,   157,   150,   150,   440,   157,   156,   157,   157,   151,
     157,   156,   157,   157,   151,   196,   150,   202,   151,   157,
     436,   156,   157,   436,   436,   157,   156,   157,   151,   157,
     156,   156,   157,   157,   197,   440,   151,   156,   157,   289,
     290,   436,   156,   436,   436,   156,   157,   156,   289,   440,
     156,   290,   156,   156,   157,   440,   156,   158,   436,   150,
     156
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   159,   160,   160,   161,   161,   162,   162,   162,   163,
     164,   164,   165,   165,   165,   165,   165,   165,   165,   165,
     165,   165,   165,   165,   165,   165,   165,   165,   165,   165,
     165,   165,   165,   165,   165,   166,   166,   167,   167,   168,
     168,   169,   169,   169,   170,   170,   171,   171,   172,   172,
     173,   173,   173,   174,   174,   175,   175,   176,   176,   176,
     177,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   179,   179,   180,   180,   180,   181,   181,   182,   182,
     183,   183,   184,   184,   185,   185,   185,   186,   186,   187,
     187,   188,   188,   188,   188,   188,   188,   188,   189,   190,
     191,   192,   193,   193,   194,   194,   195,   195,   196,   196,
     197,   197,   197,   198,   198,   199,   199,   200,   200,   201,
     201,   202,   202,   203,   203,   204,   204,   205,   205,   205,
     206,   206,   207,   207,   208,   208,   208,   208,   208,   208,
     208,   208,   208,   209,   210,   211,   212,   213,   214,   215,
     215,   216,   216,   217,   217,   217,   218,   218,   219,   219,
     220,   220,   220,   220,   221,   222,   222,   223,   223,   224,
     224,   224,   225,   225,   226,   226,   227,   227,   227,   227,
     228,   228,   228,   228,   229,   229,   230,   230,   231,   231,
     231,   232,   232,   233,   233,   234,   234,   234,   234,   234,
     234,   234,   234,   234,   234,   234,   234,   235,   236,   237,
     238,   238,   239,   240,   240,   241,   242,   242,   242,   243,
     244,   244,   245,   245,   246,   246,   246,   247,   247,   248,
     248,   249,   249,   249,   249,   249,   249,   249,   249,   249,
     249,   249,   249,   249,   249,   249,   249,   249,   250,   251,
     252,   253,   254,   255,   255,   256,   256,   257,   257,   258,
     259,   260,   261,   261,   262,   262,   263,   264,   264,   265,
     265,   266,   266,   266,   267,   267,   268,   268,   269,   269,
     269,   269,   269,   269,   269,   269,   269,   269,   269,   269,
     269,   269,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   282,   283,   283,   284,
     284,   284,   285,   285,   286,   286,   287,   287,   287,   287,
     287,   287,   287,   288,   288,   288,   288,   289,   289,   290,
     291,   292,   293,   294,   294,   295,   295,   296,   296,   296,
     297,   297,   298,   298,   299,   299,   299,   299,   300,   301,
     301,   302,   302,   303,   303,   303,   304,   304,   305,   305,
     306,   306,   306,   306,   307,   307,   307,   307,   307,   307,
     307,   308,   308,   309,   309,   310,   310,   310,   311,   311,
     312,   312,   313,   313,   313,   313,   313,   314,   315,   316,
     316,   317,   317,   318,   318,   318,   319,   319,   320,   320,
     321,   321,   321,   321,   322,   322,   323,   323,   324,   324,
     325,   325,   325,   326,   326,   327,   327,   328,   328,   328,
     328,   328,   328,   328,   328,   328,   328,   328,   328,   328,
     328,   329,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   340,   341,   341,   342,   342,   342,   343,
     343,   344,   344,   345,   345,   345,   345,   345,   345,   345,
     346,   347,   348,   349,   350,   350,   351,   351,   352,   352,
     352,   353,   353,   354,   354,   355,   355,   355,   355,   356,
     356,   357,   357,   358,   358,   358,   359,   359,   360,   360,
     361,   361,   361,   361,   361,   362,   363,   364,   364,   365,
     365,   366,   366,   366,   367,   367,   368,   368,   369,   369,
     369,   369,   369,   369,   369,   369,   369,   369,   369,   369,
     369,   369,   369,   369,   369,   369,   369,   369,   369,   369,
     369,   369,   370,   371,   372,   373,   374,   375,   376,   377,
     378,   379,   380,   381,   382,   383,   384,   385,   386,   387,
     388,   389,   390,   391,   391,   392,   392,   393,   393,   393,
     394,   394,   395,   395,   396,   396,   396,   396,   396,   396,
     396,   396,   396,   396,   396,   396,   396,   396,   396,   396,
     396,   396,   396,   396,   396,   396,   396,   397,   397,   398,
     399,   400,   401,   402,   403,   404,   405,   406,   407,   408,
     409,   409,   410,   410,   411,   411,   411,   412,   412,   413,
     413,   414,   414,   414,   414,   415,   416,   416,   417,   417,
     418,   418,   418,   419,   419,   420,   420,   421,   421,   421,
     421,   421,   421,   421,   421,   421,   421,   421,   422,   423,
     423,   424,   425,   426,   426,   427,   428,   429,   429,   430,
     430,   431,   432,   433,   433,   434,   434,   435,   435,   436,
     437,   437,   438,   439,   439,   440
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     2,     1,     2,     1,     1,     1,     1,     4,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     3,     2,     3,     2,     2,
       1,     1,     1,     1,     6,     5,     3,     2,     2,     1,
       1,     1,     1,     6,     5,     2,     1,     1,     1,     1,
       5,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     1,     1,     1,     1,     6,     5,     3,     2,
       3,     2,     2,     1,     1,     1,     1,     6,     5,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     4,     4,
       4,    16,    14,    12,     0,     1,     0,     1,     0,     1,
       0,     1,     3,     0,     1,     0,     1,     0,     1,     0,
       2,     0,     1,     3,     2,     2,     1,     1,     1,     1,
       6,     5,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     4,     6,     4,     8,    16,     3,
       2,     2,     1,     1,     1,     1,     6,     5,     2,     1,
       1,     1,     1,     1,     8,     3,     2,     2,     1,     1,
       1,     1,     6,     5,     2,     1,     1,     1,     1,     1,
       5,     6,    10,     9,     3,     2,     2,     1,     1,     1,
       1,     6,     5,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     4,     4,
       8,     7,     5,     4,     8,     4,     4,     6,    10,     4,
       3,     2,     2,     1,     1,     1,     1,     6,     5,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     4,     4,
       4,     4,     4,     4,     3,     4,     3,     4,     3,     4,
       4,     4,     4,     3,     4,     3,     4,     3,     2,     2,
       1,     1,     1,     1,     6,     5,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     3,     2,     2,     1,     1,
       1,     1,     6,     5,     2,     1,     1,     1,     1,     1,
       1,     1,     1,    16,    17,    15,    16,     2,     1,     2,
       4,     4,     6,     3,     2,     2,     1,     1,     1,     1,
       6,     5,     2,     1,     1,     1,     1,     1,     6,     3,
       2,     2,     1,     1,     1,     1,     6,     5,     2,     1,
       1,     1,     1,     1,    16,    12,    14,    13,    14,    15,
      18,     3,     2,     2,     1,     1,     1,     1,     6,     5,
       2,     1,     1,     1,     1,     1,     1,     4,     4,     3,
       2,     2,     1,     1,     1,     1,     6,     5,     2,     1,
       1,     1,     1,     1,     6,     4,     3,     2,     2,     1,
       1,     1,     1,     6,     5,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     4,     4,     4,     4,     6,     6,     6,     8,     8,
       8,     4,     3,     2,     2,     1,     1,     1,     1,     6,
       5,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       4,     4,     4,     4,     3,     2,     2,     1,     1,     1,
       1,     6,     5,     2,     1,     1,     1,     1,     1,     3,
       2,     2,     1,     1,     1,     1,     6,     5,     2,     1,
       1,     1,     1,     1,     1,     4,     8,     3,     2,     2,
       1,     1,     1,     1,     6,     5,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     4,     4,     4,     8,     4,     4,     8,     4,
       4,     4,     6,     6,     4,     4,     4,     4,     4,     4,
       4,     4,     4,     3,     2,     2,     1,     1,     1,     1,
       6,     5,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     6,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,    18,
       3,     2,     2,     1,     1,     1,     1,     6,     5,     2,
       1,     1,     1,     1,     1,    10,     3,     2,     2,     1,
       1,     1,     1,     6,     5,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,    10,    10,
       9,     4,     4,    10,     8,     6,     4,     8,     4,     5,
       3,     7,     2,     3,     1,     3,     1,     1,     1,     2,
       3,     1,     1,     3,     1,     1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 342 "vex_parse.y" /* yacc.c:1646  */
    {vex_ptr=make_vex((yyvsp[-1].llptr),(yyvsp[0].llptr));}
#line 2677 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 3:
#line 343 "vex_parse.y" /* yacc.c:1646  */
    {vex_ptr=make_vex((yyvsp[0].llptr),NULL);}
#line 2683 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 4:
#line 345 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 2689 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 5:
#line 346 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 2695 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 6:
#line 348 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_VEX_REV,(yyvsp[0].dvptr));}
#line 2701 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 7:
#line 349 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 2707 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 8:
#line 350 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 2713 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 9:
#line 354 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 2719 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 10:
#line 359 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].blptr));}
#line 2725 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 11:
#line 360 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].blptr));}
#line 2731 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 12:
#line 362 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_GLOBAL,(yyvsp[0].llptr));}
#line 2737 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 13:
#line 363 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_STATION,(yyvsp[0].llptr));}
#line 2743 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 14:
#line 364 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_MODE,(yyvsp[0].llptr));}
#line 2749 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 15:
#line 365 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_FREQ,(yyvsp[0].llptr));}
#line 2755 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 16:
#line 366 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_SCHED,(yyvsp[0].llptr));}
#line 2761 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 17:
#line 367 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_ANTENNA,(yyvsp[0].llptr));}
#line 2767 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 18:
#line 368 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_BBC,(yyvsp[0].llptr));}
#line 2773 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 19:
#line 369 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_CLOCK,(yyvsp[0].llptr));}
#line 2779 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 20:
#line 370 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_DAS,(yyvsp[0].llptr));}
#line 2785 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 21:
#line 371 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_EOP,(yyvsp[0].llptr));}
#line 2791 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 22:
#line 372 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_EXPER,(yyvsp[0].llptr));}
#line 2797 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 23:
#line 373 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_HEAD_POS,(yyvsp[0].llptr));}
#line 2803 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 24:
#line 374 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_IF,(yyvsp[0].llptr));}
#line 2809 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 25:
#line 375 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_PASS_ORDER,(yyvsp[0].llptr));}
#line 2815 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 26:
#line 376 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_PHASE_CAL_DETECT,(yyvsp[0].llptr));}
#line 2821 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 27:
#line 377 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_PROCEDURES,(yyvsp[0].llptr));}
#line 2827 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 28:
#line 378 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_ROLL,(yyvsp[0].llptr));}
#line 2833 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 29:
#line 380 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_SCHEDULING_PARAMS,(yyvsp[0].llptr));}
#line 2839 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 30:
#line 381 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_SEFD,(yyvsp[0].llptr));}
#line 2845 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 31:
#line 382 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_SITE,(yyvsp[0].llptr));}
#line 2851 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 32:
#line 383 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_SOURCE,(yyvsp[0].llptr));}
#line 2857 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 33:
#line 384 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_TAPELOG_OBS,(yyvsp[0].llptr));}
#line 2863 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 34:
#line 385 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.blptr)=make_block(B_TRACKS,(yyvsp[0].llptr));}
#line 2869 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 35:
#line 389 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 2875 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 36:
#line 390 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 2881 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 37:
#line 394 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 2887 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 38:
#line 395 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 2893 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 39:
#line 397 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 2899 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 40:
#line 398 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 2905 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 41:
#line 400 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 2911 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 42:
#line 401 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 2917 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 43:
#line 402 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 2923 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 44:
#line 404 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 2929 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 45:
#line 405 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 2935 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 46:
#line 409 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 2941 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 47:
#line 410 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 2947 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 48:
#line 412 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 2953 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 49:
#line 413 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 2959 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 50:
#line 415 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 2965 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 51:
#line 416 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 2971 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 52:
#line 417 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 2977 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 53:
#line 419 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 2983 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 54:
#line 421 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 2989 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 55:
#line 425 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 2995 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 56:
#line 426 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3001 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 57:
#line 428 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].qrptr));}
#line 3007 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 58:
#line 429 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3013 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 59:
#line 430 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3019 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 60:
#line 432 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.qrptr)=make_qref((yyvsp[-3].ival),(yyvsp[-1].sval),NULL);}
#line 3025 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 61:
#line 434 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_EXPER;}
#line 3031 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 62:
#line 435 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_SCHEDULING_PARAMS;}
#line 3037 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 63:
#line 436 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_PROCEDURES;}
#line 3043 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 64:
#line 437 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_EOP;}
#line 3049 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 65:
#line 438 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_FREQ;}
#line 3055 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 66:
#line 439 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_ANTENNA;}
#line 3061 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 67:
#line 440 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_BBC;}
#line 3067 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 68:
#line 441 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_CLOCK;}
#line 3073 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 69:
#line 442 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_CORR;}
#line 3079 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 70:
#line 443 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_DAS;}
#line 3085 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 71:
#line 444 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_HEAD_POS;}
#line 3091 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 72:
#line 445 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_PASS_ORDER;}
#line 3097 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 73:
#line 446 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_PHASE_CAL_DETECT;}
#line 3103 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 74:
#line 447 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_ROLL;}
#line 3109 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 75:
#line 448 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_IF;}
#line 3115 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 76:
#line 449 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_SEFD;}
#line 3121 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 77:
#line 450 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_SITE;}
#line 3127 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 78:
#line 451 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_SOURCE;}
#line 3133 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 79:
#line 452 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_TRACKS;}
#line 3139 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 80:
#line 453 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ival)=B_TAPELOG_OBS;}
#line 3145 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 81:
#line 455 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3151 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 82:
#line 456 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3157 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 83:
#line 458 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].qrptr));}
#line 3163 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 84:
#line 459 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3169 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 85:
#line 460 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3175 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 86:
#line 462 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.qrptr)=make_qref((yyvsp[-4].ival),(yyvsp[-2].sval),(yyvsp[-1].llptr));}
#line 3181 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 87:
#line 463 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.qrptr)=make_qref((yyvsp[-3].ival),(yyvsp[-1].sval),NULL);}
#line 3187 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 88:
#line 465 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-2].llptr),(yyvsp[0].sval));}
#line 3193 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 89:
#line 466 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].sval));}
#line 3199 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 90:
#line 470 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 3205 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 91:
#line 471 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 3211 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 92:
#line 473 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3217 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 93:
#line 474 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3223 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 94:
#line 476 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SCAN,(yyvsp[0].dfptr));}
#line 3229 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 95:
#line 477 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3235 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 96:
#line 478 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3241 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 97:
#line 481 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 3247 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 98:
#line 482 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 3253 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 99:
#line 484 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3259 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 100:
#line 485 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3265 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 101:
#line 487 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_START,(yyvsp[0].sval));}
#line 3271 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 102:
#line 488 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_MODE,(yyvsp[0].sval));}
#line 3277 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 103:
#line 489 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SOURCE,(yyvsp[0].sval));}
#line 3283 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 104:
#line 490 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_STATION,(yyvsp[0].snptr));}
#line 3289 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 105:
#line 491 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DATA_TRANSFER,(yyvsp[0].dtptr));}
#line 3295 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 106:
#line 492 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3301 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 107:
#line 493 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3307 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 108:
#line 495 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 3313 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 109:
#line 497 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 3319 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 110:
#line 499 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 3325 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 111:
#line 508 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.snptr)=make_station((yyvsp[-13].sval),(yyvsp[-11].dvptr),(yyvsp[-9].dvptr),(yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].sval),(yyvsp[-1].llptr));}
#line 3331 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 112:
#line 516 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dtptr)=make_data_transfer((yyvsp[-11].sval),(yyvsp[-9].sval),(yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[-1].sval));}
#line 3337 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 113:
#line 522 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dtptr)=make_data_transfer((yyvsp[-9].sval),(yyvsp[-7].sval),(yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr),NULL);}
#line 3343 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 114:
#line 524 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=NULL;}
#line 3349 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 115:
#line 525 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[0].dvptr);}
#line 3355 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 116:
#line 527 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=NULL;}
#line 3361 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 117:
#line 528 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[0].sval);}
#line 3367 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 118:
#line 530 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=NULL;}
#line 3373 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 119:
#line 531 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[0].sval);}
#line 3379 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 120:
#line 533 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 3385 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 121:
#line 534 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].dvptr));}
#line 3391 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 122:
#line 535 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(add_list(NULL,(yyvsp[-2].dvptr)),(yyvsp[0].dvptr));}
#line 3397 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 123:
#line 537 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=NULL;}
#line 3403 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 124:
#line 538 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[0].sval);}
#line 3409 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 125:
#line 540 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=NULL;}
#line 3415 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 126:
#line 541 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[0].sval);}
#line 3421 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 127:
#line 543 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=NULL;}
#line 3427 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 128:
#line 544 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[0].sval);}
#line 3433 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 129:
#line 546 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=make_dvalue(NULL,NULL);}
#line 3439 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 130:
#line 547 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=make_dvalue((yyvsp[-1].sval),(yyvsp[0].sval));}
#line 3445 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 131:
#line 549 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=NULL;}
#line 3451 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 132:
#line 550 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[0].sval);}
#line 3457 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 133:
#line 555 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 3463 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 134:
#line 556 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 3469 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 135:
#line 558 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3475 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 136:
#line 559 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3481 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 137:
#line 561 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 3487 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 138:
#line 562 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3493 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 139:
#line 563 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3499 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 140:
#line 566 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 3505 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 141:
#line 567 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 3511 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 142:
#line 569 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3517 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 143:
#line 570 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3523 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 144:
#line 572 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ANTENNA_DIAM,(yyvsp[0].dvptr));}
#line 3529 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 145:
#line 573 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ANTENNA_NAME,(yyvsp[0].dvptr));}
#line 3535 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 146:
#line 574 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_AXIS_TYPE,(yyvsp[0].atptr));}
#line 3541 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 147:
#line 575 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_AXIS_OFFSET,(yyvsp[0].dvptr));}
#line 3547 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 148:
#line 576 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ANTENNA_MOTION,(yyvsp[0].amptr));}
#line 3553 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 149:
#line 577 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_POINTING_SECTOR,(yyvsp[0].psptr));}
#line 3559 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 150:
#line 578 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 3565 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 151:
#line 579 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3571 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 152:
#line 580 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3577 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 153:
#line 582 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 3583 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 154:
#line 584 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 3589 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 155:
#line 587 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.atptr)=make_axis_type((yyvsp[-3].sval),(yyvsp[-1].sval));}
#line 3595 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 156:
#line 589 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 3601 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 157:
#line 594 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.amptr)=make_antenna_motion((yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 3607 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 158:
#line 603 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.psptr)=make_pointing_sector((yyvsp[-13].sval),(yyvsp[-11].sval),(yyvsp[-9].dvptr),(yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 3613 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 159:
#line 607 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 3619 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 160:
#line 608 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 3625 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 161:
#line 610 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3631 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 162:
#line 611 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3637 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 163:
#line 613 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 3643 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 164:
#line 614 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3649 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 165:
#line 615 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3655 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 166:
#line 617 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 3661 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 167:
#line 619 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 3667 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 168:
#line 621 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3673 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 169:
#line 622 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3679 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 170:
#line 624 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_BBC_ASSIGN,(yyvsp[0].baptr));}
#line 3685 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 171:
#line 625 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 3691 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 172:
#line 626 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3697 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 173:
#line 627 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3703 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 174:
#line 630 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.baptr)=make_bbc_assign((yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].sval));}
#line 3709 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 175:
#line 634 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 3715 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 176:
#line 635 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 3721 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 177:
#line 637 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3727 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 178:
#line 638 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3733 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 179:
#line 640 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 3739 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 180:
#line 641 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3745 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 181:
#line 642 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3751 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 182:
#line 645 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 3757 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 183:
#line 647 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 3763 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 184:
#line 649 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3769 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 185:
#line 650 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3775 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 186:
#line 652 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_CLOCK_EARLY,(yyvsp[0].ceptr));}
#line 3781 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 187:
#line 653 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 3787 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 188:
#line 654 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3793 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 189:
#line 655 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3799 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 190:
#line 658 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ceptr)=make_clock_early(NULL,(yyvsp[-1].dvptr),NULL,NULL);}
#line 3805 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 191:
#line 660 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ceptr)=make_clock_early((yyvsp[-3].sval),(yyvsp[-1].dvptr),NULL,NULL);}
#line 3811 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 192:
#line 662 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ceptr)=make_clock_early((yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 3817 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 193:
#line 664 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ceptr)=make_clock_early(NULL,(yyvsp[-5].dvptr),(yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 3823 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 194:
#line 668 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 3829 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 195:
#line 669 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 3835 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 196:
#line 671 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3841 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 197:
#line 672 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3847 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 198:
#line 674 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 3853 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 199:
#line 675 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3859 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 200:
#line 676 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3865 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 201:
#line 678 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 3871 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 202:
#line 680 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 3877 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 203:
#line 682 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 3883 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 204:
#line 683 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 3889 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 205:
#line 685 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_RECORD_TRANSPORT_TYPE,(yyvsp[0].sval));}
#line 3895 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 206:
#line 686 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ELECTRONICS_RACK_TYPE,(yyvsp[0].sval));}
#line 3901 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 207:
#line 687 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NUMBER_DRIVES,(yyvsp[0].dvptr));}
#line 3907 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 208:
#line 688 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_HEADSTACK,(yyvsp[0].hsptr));}
#line 3913 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 209:
#line 689 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_RECORD_DENSITY,(yyvsp[0].dvptr));}
#line 3919 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 210:
#line 690 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TAPE_LENGTH,(yyvsp[0].tlptr));}
#line 3925 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 211:
#line 692 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_RECORDING_SYSTEM_ID,(yyvsp[0].dvptr));}
#line 3931 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 212:
#line 693 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TAPE_MOTION,(yyvsp[0].tmptr));}
#line 3937 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 213:
#line 694 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TAPE_CONTROL,(yyvsp[0].sval));}
#line 3943 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 214:
#line 695 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 3949 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 215:
#line 696 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 3955 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 216:
#line 697 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 3961 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 217:
#line 699 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 3967 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 218:
#line 701 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 3973 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 219:
#line 703 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 3979 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 220:
#line 706 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.hsptr)=make_headstack((yyvsp[-5].dvptr),(yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 3985 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 221:
#line 708 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.hsptr)=make_headstack((yyvsp[-4].dvptr),NULL,(yyvsp[-1].dvptr));}
#line 3991 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 222:
#line 711 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=make_dvalue((yyvsp[-2].sval),(yyvsp[-1].sval));}
#line 3997 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 223:
#line 714 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.tlptr)=make_tape_length((yyvsp[-1].dvptr),NULL,NULL);}
#line 4003 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 224:
#line 716 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.tlptr)=make_tape_length((yyvsp[-5].dvptr),(yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 4009 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 225:
#line 718 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4015 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 226:
#line 721 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.tmptr)=make_tape_motion((yyvsp[-1].sval),NULL,NULL,NULL);}
#line 4021 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 227:
#line 723 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.tmptr)=make_tape_motion((yyvsp[-3].sval),(yyvsp[-1].dvptr),NULL,NULL);}
#line 4027 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 228:
#line 726 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.tmptr)=make_tape_motion((yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 4033 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 229:
#line 728 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4039 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 230:
#line 732 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 4045 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 231:
#line 733 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4051 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 232:
#line 735 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4057 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 233:
#line 736 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4063 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 234:
#line 738 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 4069 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 235:
#line 739 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4075 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 236:
#line 740 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4081 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 237:
#line 742 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 4087 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 238:
#line 744 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 4093 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 239:
#line 746 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4099 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 240:
#line 747 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4105 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 241:
#line 749 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TAI_UTC,(yyvsp[0].dvptr));}
#line 4111 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 242:
#line 750 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_A1_TAI,(yyvsp[0].dvptr));}
#line 4117 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 243:
#line 751 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EOP_REF_EPOCH,(yyvsp[0].sval));}
#line 4123 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 244:
#line 752 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NUM_EOP_POINTS,(yyvsp[0].dvptr));}
#line 4129 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 245:
#line 753 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EOP_INTERVAL,(yyvsp[0].dvptr));}
#line 4135 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 246:
#line 754 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_UT1_UTC,(yyvsp[0].llptr));}
#line 4141 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 247:
#line 755 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_X_WOBBLE,(yyvsp[0].llptr));}
#line 4147 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 248:
#line 756 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_Y_WOBBLE,(yyvsp[0].llptr));}
#line 4153 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 249:
#line 757 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NUT_REF_EPOCH,(yyvsp[0].sval));}
#line 4159 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 250:
#line 758 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NUM_NUT_POINTS,(yyvsp[0].dvptr));}
#line 4165 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 251:
#line 759 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NUT_INTERVAL,(yyvsp[0].dvptr));}
#line 4171 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 252:
#line 760 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DELTA_PSI,(yyvsp[0].llptr));}
#line 4177 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 253:
#line 761 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DELTA_EPS,(yyvsp[0].llptr));}
#line 4183 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 254:
#line 762 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NUT_MODEL,(yyvsp[0].sval));}
#line 4189 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 255:
#line 763 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 4195 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 256:
#line 764 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4201 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 257:
#line 765 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4207 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 258:
#line 767 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4213 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 259:
#line 769 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4219 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 260:
#line 771 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4225 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 261:
#line 773 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4231 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 262:
#line 775 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4237 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 263:
#line 777 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 4243 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 264:
#line 778 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4249 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 265:
#line 780 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 4255 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 266:
#line 781 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4261 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 267:
#line 783 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 4267 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 268:
#line 784 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4273 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 269:
#line 786 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4279 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 270:
#line 788 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4285 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 271:
#line 790 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4291 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 272:
#line 792 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 4297 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 273:
#line 793 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4303 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 274:
#line 795 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 4309 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 275:
#line 796 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4315 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 276:
#line 798 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4321 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 277:
#line 802 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 4327 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 278:
#line 803 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4333 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 279:
#line 805 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4339 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 280:
#line 806 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4345 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 281:
#line 808 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 4351 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 282:
#line 809 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4357 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 283:
#line 810 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4363 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 284:
#line 813 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 4369 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 285:
#line 814 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 4375 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 286:
#line 816 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4381 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 287:
#line 817 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4387 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 288:
#line 819 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EXPER_NUM,(yyvsp[0].dvptr));}
#line 4393 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 289:
#line 820 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EXPER_NAME,(yyvsp[0].sval));}
#line 4399 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 290:
#line 821 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EXPER_DESCRIPTION,(yyvsp[0].sval));}
#line 4405 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 291:
#line 823 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EXPER_NOMINAL_START,(yyvsp[0].sval));}
#line 4411 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 292:
#line 825 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_EXPER_NOMINAL_STOP,(yyvsp[0].sval));}
#line 4417 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 293:
#line 826 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PI_NAME,(yyvsp[0].sval));}
#line 4423 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 294:
#line 827 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PI_EMAIL,(yyvsp[0].sval));}
#line 4429 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 295:
#line 828 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_CONTACT_NAME,(yyvsp[0].sval));}
#line 4435 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 296:
#line 829 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_CONTACT_EMAIL,(yyvsp[0].sval));}
#line 4441 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 297:
#line 830 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SCHEDULER_NAME,(yyvsp[0].sval));}
#line 4447 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 298:
#line 831 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SCHEDULER_EMAIL,(yyvsp[0].sval));}
#line 4453 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 299:
#line 833 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TARGET_CORRELATOR,(yyvsp[0].sval));}
#line 4459 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 300:
#line 834 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 4465 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 301:
#line 835 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4471 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 302:
#line 836 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4477 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 303:
#line 838 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4483 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 304:
#line 840 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4489 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 305:
#line 842 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4495 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 306:
#line 844 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4501 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 307:
#line 846 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4507 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 308:
#line 848 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4513 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 309:
#line 850 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4519 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 310:
#line 852 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4525 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 311:
#line 854 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4531 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 312:
#line 856 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4537 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 313:
#line 858 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4543 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 314:
#line 860 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 4549 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 315:
#line 864 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 4555 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 316:
#line 865 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4561 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 317:
#line 867 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4567 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 318:
#line 868 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4573 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 319:
#line 870 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 4579 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 320:
#line 871 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4585 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 321:
#line 872 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4591 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 322:
#line 874 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 4597 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 323:
#line 876 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 4603 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 324:
#line 878 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4609 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 325:
#line 879 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4615 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 326:
#line 881 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_CHAN_DEF,(yyvsp[0].cdptr));}
#line 4621 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 327:
#line 882 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SAMPLE_RATE,(yyvsp[0].dvptr));}
#line 4627 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 328:
#line 883 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_BITS_PER_SAMPLE,(yyvsp[0].dvptr));}
#line 4633 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 329:
#line 884 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SWITCHING_CYCLE,(yyvsp[0].scptr));}
#line 4639 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 330:
#line 885 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 4645 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 331:
#line 886 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4651 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 332:
#line 887 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4657 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 333:
#line 896 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.cdptr)=make_chan_def((yyvsp[-13].sval),(yyvsp[-11].dvptr),(yyvsp[-9].sval),(yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].sval),(yyvsp[-1].sval),NULL);}
#line 4663 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 334:
#line 904 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.cdptr)=make_chan_def((yyvsp[-14].sval),(yyvsp[-12].dvptr),(yyvsp[-10].sval),(yyvsp[-8].dvptr),(yyvsp[-6].sval),(yyvsp[-4].sval),(yyvsp[-2].sval),(yyvsp[-1].llptr));}
#line 4669 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 335:
#line 912 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.cdptr)=make_chan_def(NULL,(yyvsp[-11].dvptr),(yyvsp[-9].sval),(yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].sval),(yyvsp[-1].sval),NULL);}
#line 4675 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 336:
#line 920 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.cdptr)=make_chan_def(NULL,(yyvsp[-12].dvptr),(yyvsp[-10].sval),(yyvsp[-8].dvptr),(yyvsp[-6].sval),(yyvsp[-4].sval),(yyvsp[-2].sval),(yyvsp[-1].llptr));}
#line 4681 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 337:
#line 922 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].dvptr));}
#line 4687 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 338:
#line 923 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].dvptr));}
#line 4693 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 339:
#line 925 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[0].dvptr);}
#line 4699 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 340:
#line 927 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4705 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 341:
#line 929 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 4711 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 342:
#line 932 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.scptr)=make_switching_cycle((yyvsp[-3].sval),(yyvsp[-1].llptr));}
#line 4717 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 343:
#line 936 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 4723 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 344:
#line 937 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4729 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 345:
#line 939 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4735 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 346:
#line 940 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4741 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 347:
#line 942 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 4747 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 348:
#line 943 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4753 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 349:
#line 944 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4759 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 350:
#line 947 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 4765 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 351:
#line 949 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 4771 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 352:
#line 951 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4777 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 353:
#line 952 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4783 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 354:
#line 954 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_HEADSTACK_POS,(yyvsp[0].hpptr));}
#line 4789 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 355:
#line 955 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 4795 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 356:
#line 956 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4801 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 357:
#line 957 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4807 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 358:
#line 960 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.hpptr)=make_headstack_pos((yyvsp[-3].dvptr),(yyvsp[-1].llptr));}
#line 4813 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 359:
#line 964 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 4819 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 360:
#line 965 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4825 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 361:
#line 967 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4831 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 362:
#line 968 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4837 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 363:
#line 970 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 4843 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 364:
#line 971 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4849 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 365:
#line 972 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4855 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 366:
#line 974 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 4861 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 367:
#line 976 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 4867 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 368:
#line 978 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4873 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 369:
#line 979 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4879 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 370:
#line 981 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_IF_DEF,(yyvsp[0].ifptr));}
#line 4885 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 371:
#line 982 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 4891 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 372:
#line 983 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4897 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 373:
#line 984 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4903 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1002 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-13].sval),(yyvsp[-11].sval),(yyvsp[-9].sval),(yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr),NULL);}
#line 4909 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1004 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-9].sval),(yyvsp[-7].sval),(yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].sval),NULL,NULL,NULL);}
#line 4915 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1006 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-11].sval),(yyvsp[-9].sval),(yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].sval),NULL,NULL,NULL);}
#line 4921 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1008 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-10].sval),(yyvsp[-8].sval),(yyvsp[-6].sval),(yyvsp[-4].dvptr),(yyvsp[-2].sval),NULL,NULL,NULL);}
#line 4927 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1010 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-11].sval),(yyvsp[-9].sval),(yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].sval),(yyvsp[-1].dvptr),NULL,NULL);}
#line 4933 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1012 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-12].sval),(yyvsp[-10].sval),(yyvsp[-8].sval),(yyvsp[-6].dvptr),(yyvsp[-4].sval),(yyvsp[-2].dvptr),NULL,NULL);}
#line 4939 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1014 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ifptr)=make_if_def((yyvsp[-15].sval),(yyvsp[-13].sval),(yyvsp[-11].sval),(yyvsp[-9].dvptr),(yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[0].sval));}
#line 4945 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1020 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 4951 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1021 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 4957 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1023 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 4963 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1025 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 4969 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1027 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 4975 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1028 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 4981 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1029 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 4987 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1032 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 4993 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1034 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 4999 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1037 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5005 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1038 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5011 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1040 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PASS_ORDER,(yyvsp[0].llptr));}
#line 5017 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1042 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_S2_GROUP_ORDER,(yyvsp[0].llptr));}
#line 5023 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 394:
#line 1043 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5029 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 395:
#line 1044 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5035 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 396:
#line 1045 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5041 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 397:
#line 1047 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 5047 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 398:
#line 1049 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 5053 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 399:
#line 1053 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 5059 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 400:
#line 1054 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 5065 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 401:
#line 1057 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5071 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 402:
#line 1058 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5077 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 403:
#line 1060 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 5083 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 404:
#line 1061 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5089 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 405:
#line 1062 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5095 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 406:
#line 1065 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 5101 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 407:
#line 1066 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 5107 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 408:
#line 1069 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5113 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 409:
#line 1070 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5119 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 410:
#line 1072 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PHASE_CAL_DETECT,(yyvsp[0].pdptr));}
#line 5125 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 411:
#line 1073 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5131 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 412:
#line 1074 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5137 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 413:
#line 1075 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5143 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 414:
#line 1078 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.pdptr)=make_phase_cal_detect((yyvsp[-3].sval),(yyvsp[-1].llptr));}
#line 5149 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 415:
#line 1080 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.pdptr)=make_phase_cal_detect((yyvsp[-1].sval),NULL);}
#line 5155 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 416:
#line 1084 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 5161 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 417:
#line 1085 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 5167 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 418:
#line 1088 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5173 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 419:
#line 1089 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5179 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 420:
#line 1091 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 5185 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 421:
#line 1092 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5191 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 422:
#line 1093 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5197 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 423:
#line 1096 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 5203 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 424:
#line 1098 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 5209 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 425:
#line 1101 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5215 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 426:
#line 1102 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5221 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 427:
#line 1105 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TAPE_CHANGE,(yyvsp[0].dvptr));}
#line 5227 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 428:
#line 1107 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_HEADSTACK_MOTION,(yyvsp[0].dvptr));}
#line 5233 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 429:
#line 1109 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NEW_SOURCE_COMMAND,(yyvsp[0].dvptr));}
#line 5239 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 430:
#line 1111 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_NEW_TAPE_SETUP,(yyvsp[0].dvptr));}
#line 5245 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 431:
#line 1113 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SETUP_ALWAYS,(yyvsp[0].saptr));}
#line 5251 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 432:
#line 1115 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PARITY_CHECK,(yyvsp[0].pcptr));}
#line 5257 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 433:
#line 1117 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TAPE_PREPASS,(yyvsp[0].tpptr));}
#line 5263 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 434:
#line 1119 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PREOB_CAL,(yyvsp[0].prptr));}
#line 5269 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 435:
#line 1121 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_MIDOB_CAL,(yyvsp[0].miptr));}
#line 5275 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 436:
#line 1123 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_POSTOB_CAL,(yyvsp[0].poptr));}
#line 5281 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 437:
#line 1125 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_PROCEDURE_NAME_PREFIX,(yyvsp[0].sval));}
#line 5287 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 438:
#line 1126 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5293 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 439:
#line 1127 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5299 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 440:
#line 1128 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5305 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 441:
#line 1130 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5311 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 442:
#line 1132 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5317 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 443:
#line 1134 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5323 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 444:
#line 1136 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5329 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 445:
#line 1139 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.saptr)=make_setup_always((yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 5335 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 446:
#line 1142 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.pcptr)=make_parity_check((yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 5341 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 447:
#line 1145 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.tpptr)=make_tape_prepass((yyvsp[-3].sval),(yyvsp[-1].dvptr));}
#line 5347 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 448:
#line 1148 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.prptr)=make_preob_cal((yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].sval));}
#line 5353 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 449:
#line 1151 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.miptr)=make_midob_cal((yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].sval));}
#line 5359 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 450:
#line 1154 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.poptr)=make_postob_cal((yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].sval));}
#line 5365 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 451:
#line 1156 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5371 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 452:
#line 1160 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 5377 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 453:
#line 1161 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 5383 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 454:
#line 1163 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5389 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 455:
#line 1164 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5395 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 456:
#line 1166 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 5401 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 457:
#line 1167 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5407 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 458:
#line 1168 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5413 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 459:
#line 1171 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 5419 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 460:
#line 1173 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 5425 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 461:
#line 1175 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5431 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 462:
#line 1176 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5437 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 463:
#line 1178 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ROLL_REINIT_PERIOD,(yyvsp[0].dvptr));}
#line 5443 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 464:
#line 1179 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ROLL_INC_PERIOD,(yyvsp[0].dvptr));}
#line 5449 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 465:
#line 1180 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ROLL,(yyvsp[0].sval));}
#line 5455 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 466:
#line 1181 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ROLL_DEF,(yyvsp[0].llptr));}
#line 5461 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 467:
#line 1182 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5467 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 468:
#line 1183 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5473 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 469:
#line 1184 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5479 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 470:
#line 1186 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5485 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 471:
#line 1188 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5491 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 472:
#line 1190 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5497 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 473:
#line 1192 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 5503 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 474:
#line 1197 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 5509 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 475:
#line 1198 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 5515 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 476:
#line 1201 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5521 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 477:
#line 1203 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5527 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 478:
#line 1205 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 5533 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 479:
#line 1206 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5539 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 480:
#line 1207 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5545 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 481:
#line 1210 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 5551 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 482:
#line 1212 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 5557 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 483:
#line 1215 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5563 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 484:
#line 1217 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5569 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 485:
#line 1219 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5575 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 486:
#line 1220 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_LITERAL,(yyvsp[0].llptr));}
#line 5581 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 487:
#line 1221 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5587 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 488:
#line 1222 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5593 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 489:
#line 1226 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 5599 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 490:
#line 1227 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 5605 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 491:
#line 1229 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5611 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 492:
#line 1230 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5617 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 493:
#line 1232 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 5623 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 494:
#line 1233 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5629 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 495:
#line 1234 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5635 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 496:
#line 1237 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 5641 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 497:
#line 1239 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 5647 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 498:
#line 1241 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5653 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 499:
#line 1242 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5659 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 500:
#line 1244 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SEFD_MODEL,(yyvsp[0].sval));}
#line 5665 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 501:
#line 1245 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SEFD,(yyvsp[0].septr));}
#line 5671 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 502:
#line 1246 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5677 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 503:
#line 1247 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5683 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 504:
#line 1248 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5689 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 505:
#line 1250 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5695 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 506:
#line 1253 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.septr)=make_sefd((yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].llptr));}
#line 5701 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 507:
#line 1257 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 5707 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 508:
#line 1258 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 5713 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 509:
#line 1260 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5719 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 510:
#line 1261 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5725 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 511:
#line 1263 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 5731 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 512:
#line 1264 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5737 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 513:
#line 1265 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5743 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 514:
#line 1268 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 5749 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 515:
#line 1269 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 5755 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 516:
#line 1271 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 5761 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 517:
#line 1272 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 5767 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 518:
#line 1274 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_TYPE,(yyvsp[0].sval));}
#line 5773 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 519:
#line 1275 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_NAME,(yyvsp[0].sval));}
#line 5779 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 520:
#line 1276 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_ID,(yyvsp[0].sval));}
#line 5785 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 521:
#line 1277 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_POSITION,(yyvsp[0].spptr));}
#line 5791 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 522:
#line 1278 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_POSITION_EPOCH,(yyvsp[0].sval));}
#line 5797 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 523:
#line 1279 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_POSITION_REF,(yyvsp[0].sval));}
#line 5803 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 524:
#line 1280 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SITE_VELOCITY,(yyvsp[0].svptr));}
#line 5809 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 525:
#line 1281 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_HORIZON_MAP_AZ,(yyvsp[0].llptr));}
#line 5815 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 526:
#line 1282 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_HORIZON_MAP_EL,(yyvsp[0].llptr));}
#line 5821 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 527:
#line 1283 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ZEN_ATMOS,(yyvsp[0].dvptr));}
#line 5827 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 528:
#line 1284 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_OCEAN_LOAD_VERT,(yyvsp[0].ovptr));}
#line 5833 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 529:
#line 1285 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_OCEAN_LOAD_HORIZ,(yyvsp[0].ohptr));}
#line 5839 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 530:
#line 1286 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_OCCUPATION_CODE,(yyvsp[0].sval));}
#line 5845 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 531:
#line 1287 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_INCLINATION,(yyvsp[0].dvptr));}
#line 5851 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 532:
#line 1288 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ECCENTRICITY,(yyvsp[0].dvptr));}
#line 5857 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 533:
#line 1289 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ARG_PERIGEE,(yyvsp[0].dvptr));}
#line 5863 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 534:
#line 1290 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ASCENDING_NODE,(yyvsp[0].dvptr));}
#line 5869 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 535:
#line 1291 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_MEAN_ANOMALY,(yyvsp[0].dvptr));}
#line 5875 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 536:
#line 1292 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SEMI_MAJOR_AXIS,(yyvsp[0].dvptr));}
#line 5881 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 537:
#line 1293 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_MEAN_MOTION,(yyvsp[0].dvptr));}
#line 5887 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 538:
#line 1294 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ORBIT_EPOCH,(yyvsp[0].sval));}
#line 5893 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 539:
#line 1295 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 5899 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 540:
#line 1296 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 5905 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 541:
#line 1297 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 5911 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 542:
#line 1299 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5917 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 543:
#line 1301 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5923 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 544:
#line 1303 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5929 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 545:
#line 1307 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.spptr)=make_site_position((yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 5935 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 546:
#line 1309 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5941 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 547:
#line 1311 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5947 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 548:
#line 1315 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.svptr)=make_site_velocity((yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 5953 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 549:
#line 1317 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 5959 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 550:
#line 1319 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 5965 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 551:
#line 1321 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5971 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 552:
#line 1325 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ovptr)=make_ocean_load_vert((yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 5977 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 553:
#line 1329 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.ohptr)=make_ocean_load_horiz((yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 5983 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 554:
#line 1331 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 5989 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 555:
#line 1333 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 5995 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 556:
#line 1335 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6001 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 557:
#line 1337 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6007 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 558:
#line 1339 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6013 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 559:
#line 1341 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6019 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 560:
#line 1343 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6025 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 561:
#line 1345 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6031 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 562:
#line 1347 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6037 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 563:
#line 1351 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 6043 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 564:
#line 1352 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 6049 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 565:
#line 1354 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 6055 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 566:
#line 1355 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 6061 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 567:
#line 1357 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 6067 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 568:
#line 1358 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 6073 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 569:
#line 1359 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 6079 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 570:
#line 1362 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 6085 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 571:
#line 1364 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 6091 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 572:
#line 1366 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 6097 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 573:
#line 1367 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 6103 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 574:
#line 1369 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SOURCE_TYPE,(yyvsp[0].llptr));}
#line 6109 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 575:
#line 1370 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SOURCE_NAME,(yyvsp[0].sval));}
#line 6115 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 576:
#line 1371 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_IAU_NAME,(yyvsp[0].sval));}
#line 6121 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 577:
#line 1372 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_RA,(yyvsp[0].sval));}
#line 6127 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 578:
#line 1373 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEC,(yyvsp[0].sval));}
#line 6133 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 579:
#line 1374 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF_COORD_FRAME,(yyvsp[0].sval));}
#line 6139 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 580:
#line 1375 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SOURCE_POSITION_REF,(yyvsp[0].sval));}
#line 6145 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 581:
#line 1376 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SOURCE_POSITION_EPOCH,(yyvsp[0].sval));}
#line 6151 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 582:
#line 1377 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_RA_RATE,(yyvsp[0].dvptr));}
#line 6157 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 583:
#line 1378 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEC_RATE,(yyvsp[0].dvptr));}
#line 6163 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 584:
#line 1379 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_VELOCITY_WRT_LSR,(yyvsp[0].dvptr));}
#line 6169 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 585:
#line 1380 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SOURCE_MODEL,(yyvsp[0].smptr));}
#line 6175 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 586:
#line 1381 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_INCLINATION,(yyvsp[0].dvptr));}
#line 6181 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 587:
#line 1382 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ECCENTRICITY,(yyvsp[0].dvptr));}
#line 6187 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 588:
#line 1383 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ARG_PERIGEE,(yyvsp[0].dvptr));}
#line 6193 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 589:
#line 1384 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ASCENDING_NODE,(yyvsp[0].dvptr));}
#line 6199 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 590:
#line 1385 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_MEAN_ANOMALY,(yyvsp[0].dvptr));}
#line 6205 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 591:
#line 1386 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_SEMI_MAJOR_AXIS,(yyvsp[0].dvptr));}
#line 6211 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 592:
#line 1387 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_MEAN_MOTION,(yyvsp[0].dvptr));}
#line 6217 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 593:
#line 1388 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_ORBIT_EPOCH,(yyvsp[0].sval));}
#line 6223 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 594:
#line 1389 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 6229 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 595:
#line 1390 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 6235 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 596:
#line 1391 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 6241 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 597:
#line 1393 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[-1].sval));}
#line 6247 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 598:
#line 1395 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(add_list(NULL,(yyvsp[-3].sval)),(yyvsp[-1].sval));}
#line 6253 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 599:
#line 1397 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6259 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 600:
#line 1399 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6265 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 601:
#line 1401 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6271 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 602:
#line 1403 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6277 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 603:
#line 1405 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6283 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 604:
#line 1407 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6289 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 605:
#line 1409 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6295 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 606:
#line 1411 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6301 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 607:
#line 1413 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6307 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 608:
#line 1416 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[-1].dvptr);}
#line 6313 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 609:
#line 1426 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.smptr)=make_source_model((yyvsp[-15].dvptr),(yyvsp[-13].sval),(yyvsp[-11].dvptr),(yyvsp[-9].dvptr),(yyvsp[-7].dvptr),(yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 6319 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 610:
#line 1430 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 6325 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 611:
#line 1431 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 6331 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 612:
#line 1434 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 6337 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 613:
#line 1435 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 6343 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 614:
#line 1437 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 6349 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 615:
#line 1438 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 6355 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 616:
#line 1439 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 6361 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 617:
#line 1443 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 6367 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 618:
#line 1445 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 6373 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 619:
#line 1448 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 6379 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 620:
#line 1449 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 6385 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 621:
#line 1451 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_VSN,(yyvsp[0].vsptr));}
#line 6391 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 622:
#line 1452 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 6397 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 623:
#line 1453 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 6403 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 624:
#line 1455 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 6409 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 625:
#line 1458 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.vsptr)=make_vsn((yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].sval),(yyvsp[-1].sval));}
#line 6415 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 626:
#line 1462 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[0].llptr);}
#line 6421 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 627:
#line 1463 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=NULL;}
#line 6427 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 628:
#line 1465 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 6433 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 629:
#line 1466 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 6439 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 630:
#line 1468 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DEF,(yyvsp[0].dfptr));}
#line 6445 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 631:
#line 1469 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 6451 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 632:
#line 1470 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 6457 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 633:
#line 1473 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-4].sval),(yyvsp[-2].llptr));}
#line 6463 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 634:
#line 1475 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dfptr)=make_def((yyvsp[-3].sval),NULL);}
#line 6469 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 635:
#line 1477 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-1].llptr),(yyvsp[0].lwptr));}
#line 6475 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 636:
#line 1478 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].lwptr));}
#line 6481 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 637:
#line 1480 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_FANIN_DEF,(yyvsp[0].fiptr));}
#line 6487 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 638:
#line 1481 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_FANOUT_DEF,(yyvsp[0].foptr));}
#line 6493 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 639:
#line 1483 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_TRACK_FRAME_FORMAT,(yyvsp[0].sval));}
#line 6499 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 640:
#line 1484 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_DATA_MODULATION,(yyvsp[0].sval));}
#line 6505 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 641:
#line 1486 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_VLBA_FRMTR_SYS_TRK,(yyvsp[0].fsptr));}
#line 6511 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 642:
#line 1488 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_VLBA_TRNSPRT_SYS_TRK,(yyvsp[0].llptr));}
#line 6517 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 643:
#line 1489 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_S2_RECORDING_MODE,(yyvsp[0].sval));}
#line 6523 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 644:
#line 1490 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_S2_DATA_SOURCE,(yyvsp[0].dsptr));}
#line 6529 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 645:
#line 1491 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_REF,(yyvsp[0].exptr));}
#line 6535 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 646:
#line 1492 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT,(yyvsp[0].sval));}
#line 6541 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 647:
#line 1493 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.lwptr)=make_lowl(T_COMMENT_TRAILING,(yyvsp[0].sval));}
#line 6547 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 648:
#line 1496 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.fiptr)=make_fanin_def((yyvsp[-7].sval),(yyvsp[-5].dvptr),(yyvsp[-3].dvptr),(yyvsp[-1].llptr));}
#line 6553 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 649:
#line 1500 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.foptr)=make_fanout_def((yyvsp[-7].sval),(yyvsp[-5].llptr),(yyvsp[-3].dvptr),(yyvsp[-1].llptr));}
#line 6559 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 650:
#line 1503 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.foptr)=make_fanout_def(NULL,(yyvsp[-5].llptr),(yyvsp[-3].dvptr),(yyvsp[-1].llptr));}
#line 6565 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 651:
#line 1505 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6571 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 652:
#line 1507 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6577 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 653:
#line 1511 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.fsptr)=make_vlba_frmtr_sys_trk((yyvsp[-7].dvptr),(yyvsp[-5].sval),(yyvsp[-3].dvptr),(yyvsp[-1].dvptr));}
#line 6583 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 654:
#line 1514 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.fsptr)=make_vlba_frmtr_sys_trk((yyvsp[-5].dvptr),(yyvsp[-3].sval),(yyvsp[-1].dvptr),NULL);}
#line 6589 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 655:
#line 1517 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(add_list(NULL,(yyvsp[-3].dvptr)),(yyvsp[-1].dvptr));}
#line 6595 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 656:
#line 1519 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.sval)=(yyvsp[-1].sval);}
#line 6601 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 657:
#line 1522 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dsptr)=make_s2_data_source((yyvsp[-5].sval),(yyvsp[-3].sval),(yyvsp[-1].sval));}
#line 6607 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 658:
#line 1524 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dsptr)=make_s2_data_source((yyvsp[-1].sval),NULL,NULL);}
#line 6613 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 659:
#line 1527 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(add_list((yyvsp[-4].llptr),(yyvsp[-2].sval)),(yyvsp[0].sval));}
#line 6619 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 660:
#line 1529 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(add_list(NULL,(yyvsp[-2].sval)),(yyvsp[0].sval));}
#line 6625 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 661:
#line 1534 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.exptr)=make_external((yyvsp[-5].sval),(yyvsp[-3].ival),(yyvsp[-1].sval));}
#line 6631 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 662:
#line 1536 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=(yyvsp[-1].llptr);}
#line 6637 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 663:
#line 1538 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=ins_list((yyvsp[-2].dvptr),(yyvsp[0].llptr));}
#line 6643 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 664:
#line 1539 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].dvptr));}
#line 6649 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 665:
#line 1541 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-2].llptr),(yyvsp[0].dvptr));}
#line 6655 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 666:
#line 1542 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].dvptr));}
#line 6661 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 667:
#line 1544 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=(yyvsp[0].dvptr);}
#line 6667 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 669:
#line 1547 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=make_dvalue((yyvsp[-1].sval),(yyvsp[0].sval));}
#line 6673 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 670:
#line 1549 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-2].llptr),(yyvsp[0].sval));}
#line 6679 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 671:
#line 1550 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].sval));}
#line 6685 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 673:
#line 1554 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list((yyvsp[-2].llptr),(yyvsp[0].dvptr));}
#line 6691 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 674:
#line 1555 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.llptr)=add_list(NULL,(yyvsp[0].dvptr));}
#line 6697 "vex_parse.c" /* yacc.c:1646  */
    break;

  case 675:
#line 1557 "vex_parse.y" /* yacc.c:1646  */
    {(yyval.dvptr)=make_dvalue((yyvsp[0].sval),NULL);}
#line 6703 "vex_parse.c" /* yacc.c:1646  */
    break;


#line 6707 "vex_parse.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 1559 "vex_parse.y" /* yacc.c:1906  */


yyerror(s)
char *s;
{
  fprintf(stderr,"%s at line %d\n",s,lines);
}


