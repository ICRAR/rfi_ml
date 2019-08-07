#ifndef __difxio_h__
#define __difxio_h__

#define DIFXIO_POL_R			0x01
#define DIFXIO_POL_L			0x02
#define DIFXIO_POL_X			0x10
#define DIFXIO_POL_Y			0x20
#define DIFXIO_POL_ERROR		0x100
#define DIFXIO_POL_RL			(DIFXIO_POL_R | DIFXIO_POL_L)
#define DIFXIO_POL_XY			(DIFXIO_POL_X | DIFXIO_POL_Y)

enum ToneSelection
{
	ToneSelectionVex = 0,		/* trust the vex file	[default] */
	ToneSelectionNone,		/* Don't pass any tones along */
	ToneSelectionEnds,		/* send along two tones at edges of the band */
	ToneSelectionAll,		/* send along all tones */
	ToneSelectionSmart,		/* like Ends, but try to stay toneGuard MHz away from band edges */
	ToneSelectionMost,		/* all except those within toneGuard */
	ToneSelectionUnknown,		/* an error condition */

	NumToneSelections		/* needs to be at end of list */
};

enum SamplingType
{
	SamplingReal = 0,
	SamplingComplex,	/* "standard" complex sampling: separate quanization of real and imag */
	SamplingComplexDSB,	/* Complex double sideband sampling */
	NumSamplingTypes	/* must remain as last entry */
};

enum DataSource
{
	DataSourceNone = 0,
	DataSourceModule,
	DataSourceFile,
	DataSourceNetwork,
	DataSourceFake,
	DataSourceMark6,
	DataSourceSharedMemory,
	DataSourceUnspecified,	/* must remain as second last entry */
	NumDataSources		/* must remain as last entry */
};

#endif // __difxio_h__
