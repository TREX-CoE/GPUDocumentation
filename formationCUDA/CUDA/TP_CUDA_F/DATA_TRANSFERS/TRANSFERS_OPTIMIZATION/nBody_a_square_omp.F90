! vim: ft=fortran :
#define M_PI 3.14159265358979323846 

module nbody_mod
  implicit none
  real :: softeningSquared = 0.001215000*0.001215000
  real :: G = 6.67259e-11
  real :: timestep = 1.0
  real :: damping = 1.0

  integer size

  real, allocatable, dimension(:) :: array_x, array_y, array_z
  real, allocatable, dimension(:) :: array_vx, array_vy, array_vz
  real, allocatable, dimension(:) :: array_mass

  contains
    subroutine update_velocity()
      integer i, j
      real :: tmpax, tmpay, tmpaz, vx, vy, vz
      real, dimension(3) :: distance
      real distanceSqr, distanceInv, val

      !$omp parallel for
      do i=1, size
        tmpax = 0.0
        tmpay = 0.0
        tmpaz = 0.0

        do j=1, size
          distance(1) = array_x(j) - array_x(i)
          distance(2) = array_y(j) - array_y(i)
          distance(3) = array_z(j) - array_z(i)

          distanceSqr = sqrt(distance(1)**2 + distance(2)**2 + distance(3)**2)+softeningSquared
          distanceInv = 1.0 / distanceSqr

          val =  G * array_mass(j) * distanceInv**3

          tmpax = tmpax + distance(1)*val
          tmpay = tmpay + distance(2)*val
          tmpaz = tmpaz + distance(3)*val
        end do
        !! optimization use of scalars inside the j loop

        vx = array_vx(i) + tmpax * timestep*damping
        vy = array_vy(i) + tmpay * timestep*damping
        vz = array_vz(i) + tmpaz * timestep*damping

        array_vx(i) = vx
        array_vy(i) = vy
        array_vz(i) = vz
      end do
    end subroutine update_velocity

    subroutine update_position()
      integer :: i
      !$omp parallel for
      do i=1, size
        array_x(i) = array_x(i) + array_vx(i) * timestep
        array_y(i) = array_y(i) + array_vy(i) * timestep
        array_z(i) = array_z(i) + array_vz(i) * timestep
      end do
    end subroutine update_position

    subroutine alloc_host_data()
      allocate(array_x(size))
      allocate(array_y(size))
      allocate(array_z(size))
      allocate(array_vx(size))
      allocate(array_vy(size))
      allocate(array_vz(size))
      allocate(array_mass(size))
    end subroutine alloc_host_data

    subroutine free_host_data()
      deallocate(array_x)
      deallocate(array_y)
      deallocate(array_z)
      deallocate(array_vx)
      deallocate(array_vy)
      deallocate(array_vz)
      deallocate(array_mass)
    end subroutine free_host_data

    subroutine init_host_data()
      integer :: i
      real :: earthmass = 5.972e24
      real :: angle, radius, vtan, anglez
      real :: x, y, z, vx, vy, vz

      !! Earth
      array_x(1) = 0.0
      array_y(1) = 0.0
      array_z(1) = 0.0
      array_vx(1) = 0.0
      array_vy(1) = 0.0
      array_vz(1) = 0.0
      array_mass(1) = earthmass

      angle = 2.0 * M_PI / real(size-1) * 0.5; !!half circle
      radius = 7000 * 1e3; !!600 + 6400) * 1e3; !!60km
      vtan  = sqrt(G*earthmass/radius);
      anglez = M_PI/8.0;

      print*, 'vtan', vtan

      !! Satelites
      do i=2, size
        x = radius * cos(angle * real(i-2)) * cos(anglez);
        y = radius * sin(angle * real(i-2));
        z = radius * cos(angle * real(i-2)) * sin(anglez);

        vx = -vtan * sin(angle * real(i-2)) * cos(anglez); !!cos(PI/2 + theta) = -sin(theta)
        vy =  vtan * cos(angle * real(i-2));               !!sin(PI/2 + theta) =  cos(theta)
        vz = -vtan * sin(angle * real(i-2)) * sin(anglez);
        array_x(i) = x
        array_y(i) = y
        array_z(i) = z
        array_vx(i) = vx
        array_vy(i) = vy
        array_vz(i) = vz
        array_mass(i) = 1.0e4
      end do
    end subroutine init_host_data

    subroutine check_res()
      integer :: i
      real :: result_x, result_y, result_z
      real :: result_vx, result_vy, result_vz
      real :: result_mass
      real :: maxvtan, minvtan, vittan

      result_x = 0.0
      result_y = 0.0
      result_z = 0.0
      result_vx = 0.0
      result_vy = 0.0
      result_vz = 0.0
      result_mass = 0.0

      do i=1, size
        result_x = result_x + array_x(i)
        result_y = result_y + array_y(i)
        result_z = result_z + array_z(i)
        result_vx = result_vx + array_vx(i)
        result_vy = result_vy + array_vy(i)
        result_vz = result_vz + array_vz(i)
        result_mass = result_mass + array_mass(i)
      end do
      print*, 'RES:', result_x, result_y, result_z, result_vx, result_vy, result_vz

      minvtan =  huge(1.0)
      maxvtan = -huge(1.0)
      do i=2, size
        vittan = sqrt(array_vx(i)**2 + array_vy(i)**2 + array_vz(i)**2 )
        minvtan=min(minvtan, vittan)
        maxvtan=max(maxvtan, vittan)
      end do
      print*, 'maxvtan', maxvtan, 'minvtan', minvtan
    end subroutine check_res

end module nbody_mod


program nBody_a_square_omp
  use nbody_mod
  use iso_c_binding
  implicit none
  interface
      function wallclock() bind(C)
        use iso_c_binding
        real(C_DOUBLE) :: wallclock
      end function wallclock
  end interface

  integer :: i, nb_steps
  real(C_DOUBLE) :: t0, t1
  character(len=32) :: arg1

  nb_steps = 200
  size = 100

  print *, 'NBODY'
  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg1)
    read(arg1, '(i9)') size ! TODO check >0
    if (command_argument_count() >= 2) then
      call get_command_argument(2, arg1)
      read(arg1, '(i9)') nb_steps ! TODO check >0
    end if
  end if


  call alloc_host_data()

  call init_host_data()

  t0 = wallclock()
  do i=1, nb_steps
    call update_velocity()
    call update_position()
  end do
  t1 = wallclock()


  write(*,'(a,i9,a,EN14.2,a,EN14.6)') 'size', size, ' sim. time (s) ', timestep*nb_steps, ' execution time ', t1-t0
  call check_res()

  call free_host_data()
end program nBody_a_square_omp
