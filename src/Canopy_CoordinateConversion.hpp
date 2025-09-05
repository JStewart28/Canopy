#ifndef CANOPY_COORDINATECONVERSION_HPP
#define CANOPY_COORDINATECONVERSION_HPP

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <memory>

#include <mpi.h>

#include <limits>

namespace Canopy
{

template <class ArrayType>
class CoordinateConversion
{
  public:
    using value_type = typename ArrayType::value_type;
    
    CoordinateConversion( const ArrayType& cartesian_center,
            const ArrayType& spherical_center
            )
        : _cartesian_center( cartesian_center )
        , _spherical_center( spherical_center )
    {
    }

    /**
     * Convert cartesian coordinates to spherical coordinates
     * Input: cart_coords = {x, y, z}
     * Output: {r, theta, phi}
     */
    KOKKOS_INLINE_FUNCTION
    ArrayType c2s( ArrayType cart_center,
                   ArrayType sph_center,
                   ArrayType cart_coords ) const
    {
        ArrayType sph;
        value_type x = cart_coords[0] - cart_center[0];
        value_type y = cart_coords[1] - cart_center[1];
        value_type z = cart_coords[2] - cart_center[2];

        value_type r = sqrt( x * x + y * y + z * z );
        value_type theta = ( r > 0.0 ) ? acos( z / r ) : 0.0;
        value_type phi = atan2( y, x );

        sph[0] = r + sph_center[0];
        sph[1] = theta + sph_center[1];
        sph[2] = phi + sph_center[2];
        return sph;
    }

    /**
     * Convert spherical coordinates to cartesian coordinates
     * Input: sph_coords = {r, theta, phi}
     * Output: {x, y, z}
     */
    KOKKOS_INLINE_FUNCTION
    ArrayType s2c( ArrayType sph_center,
                   ArrayType cart_center,
                   ArrayType sph_coords ) const
    {
        ArrayType cart;
        value_type r     = sph_coords[0] - sph_center[0];
        value_type theta = sph_coords[1] - sph_center[1];
        value_type phi   = sph_coords[2] - sph_center[2];

        value_type x = r * sin( theta ) * cos( phi );
        value_type y = r * sin( theta ) * sin( phi );
        value_type z = r * cos( theta );

        cart[0] = x + cart_center[0];
        cart[1] = y + cart_center[1];
        cart[2] = z + cart_center[2];
        return cart;
    }

    ArrayType cartesian_center() const { return _cartesian_center; }
    ArrayType spherical_center() const { return _spherical_center; }

  private:
    ArrayType _cartesian_center;
    ArrayType _spherical_center;
};

} // end namespace Canopy

#endif // CANOPY_COORDINATECONVERSION_HPP
