/***************************************************************************
 *   Copyright (C) 2009-2015 by Walter Brisken                             *
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
 * $Id: vexload.h 6904 2015-08-06 13:47:44Z WalterBrisken $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/trunk/vexdatamodel/vexload.h $
 * $LastChangedRevision: 6904 $
 * $Author: WalterBrisken $
 * $LastChangedDate: 2015-08-06 21:47:44 +0800 (Thu, 06 Aug 2015) $
 *
 *==========================================================================*/

#ifndef __VEXLOAD_H__
#define __VEXLOAD_H__

#include <string>
#include <vex_data.h>

VexData *loadVexFile(const std::string &vexFile, int *numWarnings);

#endif
