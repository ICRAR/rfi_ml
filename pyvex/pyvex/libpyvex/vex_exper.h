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
 * $Id: vex_exper.h 7208 2016-01-26 15:37:10Z WalterBrisken $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/branches/multidatastream_refactor/src/vex2difx.cpp $
 * $LastChangedRevision: 7208 $
 * $Author: WalterBrisken $
 * $LastChangedDate: 2016-01-26 23:37:10 +0800 (Tue, 26 Jan 2016) $
 *
 *==========================================================================*/

#ifndef __VEX_EXPER_H__
#define __VEX_EXPER_H__

#include <iostream>
#include "interval.h"

class VexExper : public Interval
{
public:
	VexExper() : Interval(0.0, 1000000.0) {}

	std::string name;
};

std::ostream& operator << (std::ostream &os, const VexExper &x);

#endif
