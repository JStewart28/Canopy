/****************************************************************************
 * Copyright (c) 2025 by the Canopy authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Canopy_Tags.hpp
  \brief Type tags used in Canopy
*/

#ifndef CANOPY_TAGS_HPP
#define CANOPY_TAGS_HPP

namespace Canopy
{

namespace Halo
{

//---------------------------------------------------------------------------//
// Halo direction types.
//---------------------------------------------------------------------------//
struct Vertical
{
};

struct Horizontal
{
};

//---------------------------------------------------------------------------//
// Halo data types.
//---------------------------------------------------------------------------//
struct Local
{
};

struct Multipole
{
};

struct Particle
{
};

} // end namespace Halo

} // end namespace Canopy

#endif // CANOPY_TAGS_HPP
