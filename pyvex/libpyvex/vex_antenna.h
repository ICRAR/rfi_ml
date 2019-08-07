/***************************************************************************
 *   Copyright (C) 2015-2016 by Walter Brisken & Adam Deller               *
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
 * $Id: vex_antenna.h 8036 2017-10-18 15:09:20Z WalterBrisken $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/branches/multidatastream_refactor/src/vex2difx.cpp $
 * $LastChangedRevision: 8036 $
 * $Author: WalterBrisken $
 * $LastChangedDate: 2017-10-18 23:09:20 +0800 (Wed, 18 Oct 2017) $
 *
 *==========================================================================*/

#ifndef __VEX_ANTENNA_H__
#define __VEX_ANTENNA_H__

#include <iostream>
#include <string>
#include <vector>
#include "interval.h"
#include "vex_clock.h"
#include "vex_basebanddata.h"
#include "vex_networkdata.h"

bool isVLBA(const std::string &antName);

class VexAntenna
{
public:
	VexAntenna() : x(0.0), y(0.0), z(0.0), dx(0.0), dy(0.0), dz(0.0), posEpoch(0.0), axisOffset(0.0), tcalFrequency(0) {}

	double getVexClocks(double mjd, double * coeffs) const;
	bool hasData(const Interval &timerange) const;
	void removeBasebandData(int streamId);
	bool hasVSNs() const { return !vsns.empty(); }
	bool isVLBA() const { return ::isVLBA(defName); }
	void setAntennaPolConvert(bool doConvert) { polConvert = doConvert; }

	std::string name;
	std::string defName;	// Sometimes names get changed

	double x, y, z;		// (m) antenna position in ITRF
	double dx, dy, dz;	// (m/sec) antenna velocity
	double posEpoch;	// mjd
	std::string axisType;
	double axisOffset;	// (m)
	std::vector<VexClock> clocks;
	int tcalFrequency;	// Hz
	bool polConvert;	// If true, swap polarization basis RL->XY or XY->RL

	// actual baseband data is associated with the antenna 
	std::vector<VexBasebandData> vsns;	// indexed by vsn number
	std::vector<VexBasebandData> files;	// indexed by file number
	std::vector<VexNetworkData> ports;	// indexed by stream number
};

bool usesCanonicalVDIF(const std::string &antName);

std::ostream& operator << (std::ostream &os, const VexAntenna &x);

#endif
