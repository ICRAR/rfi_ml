#include <iostream>
#include "vexload.h"

extern "C" {
#define MAP_GET_FUNCTION(name, class, instance, instance_iterator, member) \
    bool name(const class* in) { \
        if(in != instance) { \
            instance = in; \
            instance_iterator = in->member.begin(); \
        } \
        else { \
            instance_iterator++; \
        }\
        return instance_iterator != in->member.end(); \
    } \

#define VECTOR_GET_FUNCTIONS(name, class, element_class, size_getter, element_getter) \
    int name##_count(const class* in) { \
        return size_getter; \
    } \
    element_class name##_get(const class* in, int num) { \
        return element_getter(num); \
    } \

    // Exported python funcs go here

    static std::string return_string;
    
    VexData *load(const char* filename) {
        int junk;
        return loadVexFile(filename, &junk);
    }

    void free_vex(VexData* data) {
        delete data;
    }
    
    const char* get_directory(const VexData* data) {
        return_string = data->getDirectory();
        return return_string.c_str();
    }

    int get_polarisations(const VexData* data) {
        return data->getPolarizations();
    }

    const VexExper* get_exper(const VexData* data) {
        return data->getExper();
    }


	// *****************************************************************
	// * Sources
	// *****************************************************************

    VECTOR_GET_FUNCTIONS(source, VexData, const VexSource*, in->nSource(), in->getSource)

	const char* source_def_name(const VexSource* source) {
	    return_string = source->defName;
	    return return_string.c_str();
	}

	double source_ra(const VexSource* source) {
	    return source->ra;
	}

	double source_dec(const VexSource* source) {
	    return source->dec;
	}

	char source_cal_code(const VexSource* source) {
	    return source->calCode;
	}

	int source_names_count(const VexSource* source) {
	    return source->sourceNames.size();
	}

	const char* source_names_get(const VexSource* source, int index) {
	    return_string = source->sourceNames[index];
	    return return_string.c_str();
	}

	// *****************************************************************
	// * Interval
	// *****************************************************************

	double interval_start(const Interval* interval) {
	    return interval->mjdStart;
	}

	double interval_stop(const Interval* interval) {
	    return interval->mjdStop;
	}

	// *****************************************************************
	// * Scan
	// *****************************************************************

    VECTOR_GET_FUNCTIONS(vscan, VexData, const VexScan*, in->nScan(), in->getScan)

	const char* scan_def_name(const VexScan* scan) {
	    return_string = scan->defName;
	    return return_string.c_str();
	}

	const char* scan_intent(const VexScan* scan) {
	    return_string = scan->intent;
	    return return_string.c_str();
	}

	const char* scan_mode_def_name(const VexScan* scan) {
	    return_string = scan->modeDefName;
	    return return_string.c_str();
	}

	const char* scan_source_def_name(const VexScan* scan) {
	    return_string = scan->sourceDefName;
	    return return_string.c_str();
	}

	double scan_size(const VexScan* scan) {
	    return scan->size;
	}

	double scan_mjd_vex(const VexScan* scan) {
	    return scan->mjdVex;
	}

    static const VexScan* current_scan_stations = nullptr;
    static std::map<std::string, Interval>::const_iterator current_scan_stations_iterator;

    MAP_GET_FUNCTION(scan_stations_get, VexScan, current_scan_stations, current_scan_stations_iterator, stations)

	const char* scan_stations_ant() {
	    return_string = current_scan_stations_iterator->first;
	    return return_string.c_str();
	}

	const Interval* scan_stations_interval() {
	    return &current_scan_stations_iterator->second;
	}

    static const VexScan* current_scan_record_enable = nullptr;
    static std::map<std::string, bool>::const_iterator current_scan_record_enable_iterator;

    MAP_GET_FUNCTION(scan_record_enable_get, VexScan, current_scan_record_enable, current_scan_record_enable_iterator, recordEnable)

   	const char* scan_record_enable_ant() {
   	    return_string = current_scan_record_enable_iterator->first;
   	    return return_string.c_str();
   	}

   	bool scan_record_enable() {
   	    return current_scan_record_enable_iterator->second;
   	}

    // *****************************************************************
    // * Subband
    // *****************************************************************

    double subband_freq(const VexSubband* subband) {
        return subband->freq;
    }

    double subband_bandwidth(const VexSubband* subband) {
        return subband->bandwidth;
    }

    char subband_side_band(const VexSubband* subband) {
        return subband->sideBand;
    }

    char subband_pol(const VexSubband* subband) {
        return subband->pol;
    }

    // *****************************************************************
    // * IF
    // *****************************************************************

    const char* if_name(const VexIF* in) {
        return_string = in->name;
        return return_string.c_str();
    }

    double if_sslo(const VexIF* in) {
        return in->ifSSLO;
    }

    char if_side_band(const VexIF* in) {
        return in->ifSideBand;
    }

    char if_pol(const VexIF* in) {
        return in->pol;
    }

    float if_phase_cal_interval_mhz(const VexIF* in) {
        return in->phaseCalIntervalMHz;
    }

    float if_phase_cal_base_mhz(const VexIF* in) {
        return in->phaseCalBaseMHz;
    }

    const char* if_comment(const VexIF* in) {
        return_string = in->comment;
        return return_string.c_str();
    }

    const char* if_band_name(const VexIF* in) {
        return_string = in->bandName();
        return return_string.c_str();
    }

    const char* if_vlba_band_name(const VexIF* in) {
        return_string = in->VLBABandName();
        return return_string.c_str();
    }

    double if_lower_edge_freq(const VexIF* in) {
        return in->getLowerEdgeFreq();
    }

    // *****************************************************************
    // * Channel
    // *****************************************************************

    int channel_record(const VexChannel* in) {
        return in->recordChan;
    }

    int channel_subband_id(const VexChannel* in) {
        return in->subbandId;
    }

    const char* channel_if_name(const VexChannel* in) {
        return_string = in->ifName;
        return return_string.c_str();
    }

    double channel_bbc_freq(const VexChannel* in) {
        return in->bbcFreq;
    }

    double channel_bbc_bandwidth(const VexChannel* in) {
        return in->bbcBandwidth;
    }

    char channel_bbc_side_band(const VexChannel* in) {
        return in->bbcSideBand;
    }

    const char* channel_name(const VexChannel* in) {
        return_string = in->name;
        return return_string.c_str();
    }

    const char* channel_bbc_name(const VexChannel* in) {
        return_string = in->bbcName;
        return return_string.c_str();
    }

    VECTOR_GET_FUNCTIONS(channel_tone, VexChannel, unsigned int, in->tones.size(), in->tones.at)

    int channel_thread_id(const VexChannel* in) {
        return in->threadId;
    }

    // *****************************************************************
    // * Stream
    // *****************************************************************

    double stream_sample_rate(const VexStream* in) {
        return in->sampRate;
    }

    unsigned int stream_n_bit(const VexStream* in) {
        return in->nBit;
    }

    unsigned int stream_n_record_chan(const VexStream* in) {
        return in->nRecordChan;
    }

    unsigned int stream_n_thread(const VexStream* in) {
        return in->nThread;
    }

    unsigned int stream_fanout(const VexStream* in) {
        return in->fanout;
    }

    unsigned int stream_v_dif_frame_size(const VexStream* in) {
        return in->VDIFFrameSize;
    }

    bool stream_single_thread(const VexStream* in) {
        return in->singleThread;
    }

    VECTOR_GET_FUNCTIONS(stream_thread, VexStream, int, in->threads.size(), in->threads.at)

    int stream_data_format(const VexStream* in) {
        return (int)in->format;
    }

    int stream_data_sampling(const VexStream* in) {
        return (int)in->dataSampling;
    }

    int stream_data_source(const VexStream* in) {
        return (int)in->dataSource;
    }

    double stream_difx_tsys(const VexStream* in) {
        return in->difxTsys;
    }

    // *****************************************************************
    // * Setup
    // *****************************************************************

    VECTOR_GET_FUNCTIONS(mode, VexData, const VexMode*, in->nMode(), in->getMode)

    static const VexSetup* current_setup_ifs = nullptr;
    static std::map<std::string, VexIF>::const_iterator current_setup_ifs_iterator;

    MAP_GET_FUNCTION(setup_ifs_get, VexSetup, current_setup_ifs, current_setup_ifs_iterator, ifs)

    const char* setup_ifs_get_name() {
        return_string = current_setup_ifs_iterator->first;
        return return_string.c_str();
    }

    const VexIF* setup_ifs_get_if() {
        return &current_setup_ifs_iterator->second;
    }

    VECTOR_GET_FUNCTIONS(setup_channel, VexSetup, const VexChannel*, in->channels.size(), &in->channels.at)

    VECTOR_GET_FUNCTIONS(setup_stream, VexSetup, const VexStream*, in->streams.size(), &in->streams.at)

    // *****************************************************************
    // * Mode
    // *****************************************************************

    const char* mode_def_name(const VexMode* mode) {
        return_string = mode->defName;
        return return_string.c_str();
    }

    static const VexMode* current_mode_setup = nullptr;
    static std::map<std::string, VexSetup>::const_iterator current_mode_setup_iterator;

    MAP_GET_FUNCTION(mode_setups_get, VexMode, current_mode_setup, current_mode_setup_iterator, setups)

    const char* mode_setups_get_ant() {
        return_string = current_mode_setup_iterator->first;
        return return_string.c_str();
    }

    const VexSetup* mode_setups_get_setup() {
        return &current_mode_setup_iterator->second;
    }

    VECTOR_GET_FUNCTIONS(mode_subband, VexMode, const VexSubband*, in->subbands.size(), &in->subbands.at)
    VECTOR_GET_FUNCTIONS(mode_zoomband, VexMode, const VexSubband*, in->zoombands.size(), &in->zoombands.at)

    // *****************************************************************
    // * Clock
    // *****************************************************************

    double clock_mjd(VexClock* clock) {
        return clock->mjdStart;
    }

    double clock_offset(VexClock* clock) {
        return clock->offset;
    }

    double clock_rate(VexClock* clock) {
        return clock->rate;
    }

    double clock_offset_epoch(VexClock* clock) {
        return clock->offset_epoch;
    }

    // *****************************************************************
    // * BasebandData
    // *****************************************************************

    const char* baseband_data_filename(VexBasebandData* data) {
        return_string = data->filename;
        return return_string.c_str();
    }

    int baseband_recorder_id(VexBasebandData* data) {
        return data->recorderId;
    }

    int baseband_stream_id(VexBasebandData* data) {
        return data->streamId;
    }

    // *****************************************************************
    // * NetworkData
    // *****************************************************************

    const char* network_data_port(VexNetworkData* data) {
        return_string = data->networkPort;
        return return_string.c_str();
    }

    // *****************************************************************
    // * Eop
    // *****************************************************************

    VECTOR_GET_FUNCTIONS(eop, VexData, const VexEOP*, in->nEOP(), in->getEOP)

    double eop_mjd(VexEOP* eop) {
        return eop->mjd;
    }

    double eop_tai_utc(VexEOP* eop) {
        return eop->tai_utc;
    }

    double eop_ut1_utc(VexEOP* eop) {
        return eop->ut1_utc;
    }

    double eop_x_pole(VexEOP* eop) {
        return eop->xPole;
    }

    double eop_y_pole(VexEOP* eop) {
        return eop->yPole;
    }

    // *****************************************************************
    // * Antenna
    // *****************************************************************

    VECTOR_GET_FUNCTIONS(antenna, VexData, const VexAntenna*, in->nAntenna(), in->getAntenna)

    const char* antenna_name(VexAntenna* ant) {
        return_string = ant->name;
        return return_string.c_str();
    }

    const char* antenna_def_name(VexAntenna* ant) {
        return_string = ant->defName;
        return return_string.c_str();
    }

    double antenna_x(VexAntenna* ant) {
        return ant->x;
    }

    double antenna_y(VexAntenna* ant) {
        return ant->y;
    }

    double antenna_z(VexAntenna* ant) {
        return ant->z;
    }

    double antenna_dx(VexAntenna* ant) {
        return ant->dx;
    }

    double antenna_dy(VexAntenna* ant) {
        return ant->dy;
    }

    double antenna_dz(VexAntenna* ant) {
        return ant->dz;
    }

    double antenna_pos_epoch(VexAntenna* ant) {
        return ant->posEpoch;
    }

    const char* antenna_axis_type(VexAntenna* ant) {
        return_string = ant->axisType;
        return return_string.c_str();
    }

    double antenna_axis_offset(VexAntenna* ant) {
        return ant->axisOffset;
    }

    VECTOR_GET_FUNCTIONS(antenna_clock, VexAntenna, const VexClock*, in->clocks.size(), &in->clocks.at)

    int antenna_tcal_frequency(VexAntenna* ant) {
        return ant->tcalFrequency;
    }

    bool antenna_pol_convert(VexAntenna* ant) {
        return ant->polConvert;
    }

    VECTOR_GET_FUNCTIONS(antenna_vsn, VexAntenna, const VexBasebandData*, in->vsns.size(), &in->vsns.at)

    VECTOR_GET_FUNCTIONS(antenna_file, VexAntenna, const VexBasebandData*, in->files.size(), &in->files.at)

    VECTOR_GET_FUNCTIONS(antenna_port, VexAntenna, const VexNetworkData*, in->ports.size(), &in->ports.at)

    // *****************************************************************
    // * Exper
    // *****************************************************************

    const char* exper_name(VexExper* exper) {
        return_string = exper->name;
        return return_string.c_str();
    }
}
