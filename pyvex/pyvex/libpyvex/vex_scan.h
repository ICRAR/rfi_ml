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
 * $Id: vex_scan.h 7343 2016-06-14 16:29:17Z WalterBrisken $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/branches/multidatastream_refactor/src/vex2difx.cpp $
 * $LastChangedRevision: 7343 $
 * $Author: WalterBrisken $
 * $LastChangedDate: 2016-06-15 00:29:17 +0800 (Wed, 15 Jun 2016) $
 *
 *==========================================================================*/

#ifndef __VEX_SCAN_H__
#define __VEX_SCAN_H__

#include <ostream>
#include <string>
#include <map>
#include "interval.h"

class VexScan : public Interval
{
public:
	std::string defName;				// name of this scan
	std::string intent;				// intent of this scan

	std::string modeDefName;
	std::string sourceDefName;	
	std::map<std::string,Interval> stations;
	std::map<std::string,bool> recordEnable;	// This is true if the drive number is non-zero
	double size;					// [bytes] approx. correlated size
	double mjdVex;					// The start time listed in the vex file

	VexScan(): size(0), mjdVex(0.0) {};
	const Interval *getAntennaInterval(const std::string &antName) const;
	bool getRecordEnable(const std::string &antName) const;
};

std::ostream& operator << (std::ostream &os, const VexScan &x);

#endif
