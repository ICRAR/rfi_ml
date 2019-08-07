/***************************************************************************
 *   Copyright (C) 2015-2017 by Walter Brisken & Adam Deller               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
/*===========================================================================
 * SVN properties (DO NOT CHANGE)
 *
 * $Id: vex_setup.cpp 8599 2018-11-30 13:22:58Z JanWagner $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/branches/multidatastream_refactor/src/vex2difx.cpp $
 * $LastChangedRevision: 8599 $
 * $Author: JanWagner $
 * $LastChangedDate: 2018-11-30 21:22:58 +0800 (Fri, 30 Nov 2018) $
 *
 *==========================================================================*/

#include <algorithm>
#include <set>
#include "vex_setup.h"

float VexSetup::phaseCalIntervalMHz() const
{
	float p;
	float pc = 0;

	for(std::map<std::string,VexIF>::const_iterator it = ifs.begin(); it != ifs.end(); ++it)
	{
		p = it->second.phaseCalIntervalMHz;
		if(p > 0 && (p < pc || pc == 0))
		{
			pc = p;
		}
	}

	return pc;
}

float VexSetup::phaseCalBaseMHz() const
{
	float pb;
	float pcb = 0;

	for(std::map<std::string,VexIF>::const_iterator it = ifs.begin(); it != ifs.end(); ++it)
	{
		pb = it->second.phaseCalBaseMHz;
		if(pb > 0 && (pb < pcb || pcb == 0))
		{
			pcb = pb;
		}
	}

	return pcb;
}

const VexIF *VexSetup::getIF(const std::string &ifName) const
{
	for(std::map<std::string,VexIF>::const_iterator it = ifs.begin(); it != ifs.end(); ++it)
	{
		if(it->second.name == ifName)
		{
			return &it->second;
		}
	}

	return 0;
}

double VexSetup::firstTuningForIF(const std::string &ifName) const	// return Hz
{
	double tune = 0.0;	// [Hz]
	std::string chanName;

	for(std::vector<VexChannel>::const_iterator ch=channels.begin(); ch != channels.end(); ++ch)
	{
		if(ch->ifName == ifName && (chanName == "" || ch->name < chanName))
		{
			chanName = ch->name;
			tune = ch->bbcFreq;
		}
	}

	return tune;
}

double VexSetup::dataRateMbps() const
{
	double rate = 0;	// [Mbps]

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		rate += it->dataRateMbps();
	}

	return rate;
}

void VexSetup::sortChannels()
{
	sort(channels.begin(), channels.end());
}

bool VexSetup::hasUniqueRecordChans() const
{
	std::set<int> ids;

	for(std::vector<VexChannel>::const_iterator it = channels.begin(); it != channels.end(); ++it)
	{
		ids.insert(it->recordChan);
	}

	return (ids.size() == channels.size());
}

void VexSetup::assignRecordChans()
{
	int id = 0;

	sort(channels.begin(), channels.end());
	for(std::vector<VexChannel>::iterator it = channels.begin(); it != channels.end(); ++it)
	{
		it->recordChan = id++;
	}
}

void VexSetup::setPhaseCalInterval(float phaseCalIntervalMHz)
{
	// change IF phase cal values
	for(std::map<std::string,VexIF>::iterator it = ifs.begin(); it != ifs.end(); ++it)
	{
		it->second.phaseCalIntervalMHz = phaseCalIntervalMHz;
	}

	// weed out unwanted tones
	for(std::vector<VexChannel>::iterator it = channels.begin(); it != channels.end(); ++it)
	{
		if(phaseCalIntervalMHz <= 0)
		{
			it->tones.clear();
		}
		else
		{
			for(std::vector<unsigned int>::iterator tit = it->tones.begin(); tit != it->tones.end(); )
			{
				float toneFreq = (*tit) * phaseCalIntervalMHz + phaseCalBaseMHz();
				if (toneFreq > it->bbcBandwidth)
				{
					tit = it->tones.erase(tit);
				}
				else
				{
					++tit;
				}
			}
		}
	}
}

void VexSetup::setPhaseCalBase(float phaseCalBaseMHz)
{
	// change IF phase cal values
	for(std::map<std::string,VexIF>::iterator it = ifs.begin(); it != ifs.end(); ++it)
	{
		it->second.phaseCalBaseMHz = phaseCalBaseMHz;
	}

	// weed out unwanted tones
	for(std::vector<VexChannel>::iterator it = channels.begin(); it != channels.end(); ++it)
	{
		if(phaseCalBaseMHz < 0)
		{
			it->tones.clear();
		}
		else
		{
			for(std::vector<unsigned int>::iterator tit = it->tones.begin(); tit != it->tones.end(); )
			{
				float toneFreq = (*tit) * phaseCalIntervalMHz() + phaseCalBaseMHz;
				if (toneFreq > it->bbcBandwidth)
				{
					tit = it->tones.erase(tit);
				}
				else
				{
					++tit;
				}
			}
		}
	}
}

void VexSetup::selectTones(enum ToneSelection selection, double guardBandMHz)
{
	for(std::vector<VexChannel>::iterator it = channels.begin(); it != channels.end(); ++it)
	{
		const VexIF *vif = getIF(it->ifName);
		if (vif)
		{
			it->selectTones(vif->phaseCalIntervalMHz, vif->phaseCalBaseMHz, selection, guardBandMHz);
		}
	}
}

size_t VexSetup::nRecordChan() const
{
	size_t rc = 0;

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		rc += it->nRecordChan;
	}

	return rc;
}

bool VexSetup::usesFormat(enum VexStream::DataFormat format) const
{
	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		if(it->format == format)
		{
			return true;
		}
	}

	return false;
}

unsigned int VexSetup::getBits() const
{
	unsigned int b1, b2;
	bool first = true;

	b1 = getMinBits();
	b2 = getMaxBits();

	if(b1 != b2)
	{
		if(first)
		{
			first = false;
			std::cerr << "Warning: VexSetup::getBits(): different number of bits on different datastreams for one antenna." << std::endl;
		}
	}

	return b2;
}

unsigned int VexSetup::getMinBits() const
{
	unsigned int nBit = 0;

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		if(nBit == 0 && it->nBit > 0)
		{
			nBit = it->nBit;
		}
		else if(it->nBit < nBit)
		{
			nBit = it->nBit;
		}
	}

	return nBit;
}

unsigned int VexSetup::getMaxBits() const
{
	unsigned int nBit = 0;

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		if(it->nBit > nBit)
		{
			nBit = it->nBit;
		}
	}

	return nBit;
}

/* returns lowest sample rate in samples per second */
double VexSetup::getLowestSampleRate() const
{
	double sr = 0.0;

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		if(sr == 0.0 && it->sampRate > 0.0)
		{
			sr = it->sampRate;
		}
		else if(it->sampRate < sr)
		{
			sr = it->sampRate;
		}
	}

	return sr;
}

/* returns highest sample rate in samples per second */
double VexSetup::getHighestSampleRate() const
{
	double sr = 0.0;

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		if(it->sampRate > sr)
		{
			sr = it->sampRate;
		}
	}

	return sr;
}

/* returns average sample rate in samples per second */
double VexSetup::getAverageSampleRate() const
{
	double sr = 0.0;
	int n = 0;

	for(std::vector<VexStream>::const_iterator it = streams.begin(); it != streams.end(); ++it)
	{
		if(it->sampRate > 0.0)
		{
			sr += it->sampRate;
			++n;
		}
	}

	if(n > 0)
	{
		sr /= n;
	}

	return sr;
}

bool VexSetup::hasDuplicateSubbands() const
{
	unsigned int n;

	n = channels.size();

	if(n > 1)
	{
		for(unsigned int i = 1; i < n; ++i)
		{
			if(channels[i].subbandId < 0)
			{
				continue;
			}
			for(unsigned int j = 0; j < i; ++j)
			{
				if(channels[i].subbandId == channels[j].subbandId)
				{
					return true;
				}
			}
		}
	}

	return false;
}

/* returns bit map using same values as difxio does */
int VexSetup::getPolarizations() const
{
	int rv = 0;

	for(std::map<std::string,VexIF>::const_iterator it = ifs.begin(); it != ifs.end(); ++it)
	{
		switch(it->second.pol)
		{
		case 'R':
			rv |= 0x01;
			break;
		case 'L':
			rv |= 0x02;
			break;
		case 'X':
			rv |= 0x10;
			break;
		case 'Y':
			rv |= 0x20;
			break;
		default:
			rv |= 0x100;	// Error/Unknown bit
		}
	}

	return rv;
}

std::ostream& operator << (std::ostream &os, const VexSetup &x)
{
	os << "   Setup:" << std::endl;
	for(std::vector<VexChannel>::const_iterator it = x.channels.begin(); it != x.channels.end(); ++it)
	{
		os << "    Channel: " << *it << std::endl;
	}
	for(std::map<std::string,VexIF>::const_iterator it = x.ifs.begin(); it != x.ifs.end(); ++it)
	{
		os << "    IF: " << it->first << " " << it->second << std::endl;
	}
	for(std::vector<VexStream>::const_iterator it = x.streams.begin(); it != x.streams.end(); ++it)
	{
		os << "    Datastream: " << *it << std::endl;
	}

	return os;
}
