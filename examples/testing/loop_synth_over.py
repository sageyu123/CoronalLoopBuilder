import sys
sys.path.insert(1, '/home/saber/rad_transfer')

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, spherical_to_cartesian as stc
import sunpy.map
import astropy.constants as const

# Import synthetic image manipulation tools
import yt
# noinspection PyUnresolvedReferences
from utils.proj_imag import SyntheticFilterImage as synt_img
# noinspection PyUnresolvedReferences
from emission_models import uv, xrt
import pickle

from CoronalLoopBuilder.builder import CoronalLoopBuilder, circle_3d

def circle3d(theta = 90, phi = 0, r = 10 * u.Mm, x0 = 0, y0 = 0, z0 = 0, **kwargs):

    if 'pkl' in kwargs:
        with open(kwargs.get('pkl'), 'rb') as f:
            dims = pickle.load(f)

            r = dims['radius']
            theta = dims['el'].to(u.rad).value
            phi = dims['az'].to(u.rad).value

            f.close()
    else:
        # Set the loop parameters using the provided values or default values
        r = kwargs.get('radius', DEFAULT_RADIUS)
        theta = theta.to(u.rad).value
        phi = phi.to(u.rad).value

    # Normal vector
    # (Normal to the surface of the sun)
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # Arbitrary vector (z-axis)
    z = np.array([0, 0, 1])

    # Orthogonal vector in the plane of the circle
    # Orthogonal to the flare loop
    o = np.cross(n, z)
    o /= np.linalg.norm(o)  # Normalize

    # Another orthogonal vector in the plane
    v = np.cross(n, o)
    v /= np.linalg.norm(v)  # Normalize

    # # Parametric equations
    # x = x0 + r * np.cos(t) * o[0] + r * np.sin(t) * v[0]
    # y = y0 + r * np.cos(t) * o[1] + r * np.sin(t) * v[1]
    # z = z0 + r * np.cos(t) * o[2] + r * np.sin(t) * v[2]
    #
    # return x, y, z

    return n, o, v

# Method to create synthetic map of MHD data from rad_transfer
def synthmap_plot(img_path, fig, normvector=None, northvector=None, comp=False):
    # Load aia image map
    if normvector is None:
        normvector = [1, 0, 0]
    if northvector is None:
        northvector = [0, 0, 1]

    img = sunpy.map.Map(img_path)

    # Crop image map
    bl = SkyCoord(850 * u.arcsec, -330 * u.arcsec, frame=img.coordinate_frame)
    res = 250 * u.arcsec
    img = img.submap(bottom_left=bl, width=res, height=res)

    # Load and crop MHD file
    downs_file_path = '/home/saber/rad_transfer/datacubes/subs_3_flarecs-id_0012.h5'
    subs_ds = yt.load(downs_file_path)  # , hint='AthenaDataset', units_override=units_override)
    cut_box = subs_ds.region(center=[0.0, 0.5, 0.0], left_edge=[-0.5, 0.016, -0.25], right_edge=[0.5, 1.0, 0.25])

    instr = 'aia'  # keywords: 'aia' or 'xrt'
    channel = 131
    # channel = int(img.wavelength.value)
    timestep = downs_file_path[-7:-3]
    fname = os.getcwd() + "/" + str(instr) + "_" + str(channel) + '_' + timestep

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    cusp_submap = img
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    aia_synthetic = synt_img(cut_box, instr, channel)
    # Match parameters of the synthetic image to observed one
    samp_resolution = cusp_submap.data.shape[0]
    obs_scale = [cusp_submap.scale.axis1, cusp_submap.scale.axis2] * (u.arcsec / u.pixel)
    reference_pixel = u.Quantity([cusp_submap.reference_pixel[0].value,
                                  cusp_submap.reference_pixel[1].value], u.pixel)
    reference_coord = cusp_submap.reference_coordinate

    img_tilt = 0 * u.deg

    # Dynamic synth plot settings
    synth_plot_settings = {'resolution': samp_resolution,
                           'vmin': 1e-15,
                           'vmax': 1e6,
                           'cmap': 'inferno',
                           'logscale': True,
                          }
    synth_view_settings = {'normal_vector': normvector,  # Line of sight - changes 'orientation' of projection
                           'north_vector': northvector}  # rotates projection in xy

    aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
                                view_settings=synth_view_settings,
                                image_shift=[0, 0],                 # move the bottom center of the flare in [x,y]
                                bkg_fill=np.min(img.data))

    # Hardcoded synth plot settings
    # synth_plot_settings = {'resolution': samp_resolution}
    # synth_view_settings = {'normal_vector': [0, 0, 0.1],  # Line of sight - changes 'orientation' of projection
    #                        'north_vector': [np.sin(img_tilt).value, np.cos(img_tilt).value,
    #                                         0.0]}  # rotates projection in xy
    #
    # aia_synthetic.proj_and_imag(plot_settings=synth_plot_settings,
    #                             view_settings=synth_view_settings,
    #                             image_shift=[0, 0],
    #                             bkg_fill=np.min(img.data))

    # Import scale from an AIA image:
    synth_map = aia_synthetic.make_synthetic_map(obstime='2013-10-28',
                                                 observer='earth',
                                                 detector='Synthetic AIA',
                                                 scale=obs_scale,
                                                 reference_coord=reference_coord,
                                                 reference_pixel=reference_pixel)  # .rotate(angle=0.0 * u.deg)

    if comp:
        synth_map = sunpy.map.Map(synth_map, vmin=1e-5, vmax=8e1, cmap='inferno')
        comp = sunpy.map.Map(synth_map, cusp_submap, composite=True)
        comp.set_alpha(1, 0.4)
        ax = fig.add_subplot(projection=comp.get_map(0))
        comp.plot(axes=ax)
    else:
        ax = fig.add_subplot(projection=synth_map)
        synth_map.plot(axes=ax, vmin=1e-5, vmax=8e1, cmap='inferno')

    return ax

def calc_vect(radius = const.R_sun, height = 10 * u.Mm, theta0=0 * u.deg, phi0=0 * u.deg, el=90 * u.deg, az=0 * u.deg,
              samples_num=100, **kwargs):

    if 'pkl' in kwargs:
        with open(kwargs.get('pkl'), 'rb') as f:
            dims = pickle.load(f)
            print(f'Loop dimensions loaded:{dims}')
            radius = dims['radius']
            height = dims['height']
            phi0 = dims['phi0']
            theta0 = dims['theta0']
            el = dims['el']
            az = dims['az']
            f.close()
    else:
        # Set the loop parameters using the provided values or default values
        radius = kwargs.get('radius', DEFAULT_RADIUS)
        height = kwargs.get('height', DEFAULT_HEIGHT)
        phi0 = kwargs.get('phi0', DEFAULT_PHI0)
        theta0 = kwargs.get('theta0', DEFAULT_THETA0)
        el = kwargs.get('el', DEFAULT_EL)
        az = kwargs.get('az', DEFAULT_AZ)

    r_1 = const.R_sun

    r0 = r_1 + height
    x0 = u.Quantity(0 * u.cm)
    y0 = u.Quantity(0 * u.cm)
    z0 = r0.to(u.cm)

    theta = el.to(u.rad).value  # np.pi / 2  # Elevation angle
    phi = az.to(u.rad).value  # np.pi / 4  # Azimuth angle
    t = np.linspace(0, 2 * np.pi, int(samples_num))  # Parameter t

    dx, dy, dz = circle_3d(0 , 0, 0, radius, theta, phi, t)

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rdiff = r - r_1
    rsort = np.argmin(np.abs(rdiff))
    if rdiff[rsort] + rdiff[rsort + 1] < 0:
        rsort += 1
    r = np.roll(r, -rsort)
    dx = np.roll(dx, -rsort)
    dy = np.roll(dy, -rsort)
    dz = np.roll(dz, -rsort)

    i_r = np.where(r > r_1)

    dx = dx[i_r]
    dy = dy[i_r]
    dz = dz[i_r]

    # Calculate the length of the loop based on the angle between the start and end points.
    # Define the vectors v1 and v2
    v1 = np.array([dx[0].value, dy[0].value, dz[0].value]) * dx[0].unit
    v2 = np.array([dx[-1].value, dy[-1].value, dz[-1].value]) * dx[0].unit

    # Use the cross product to determine the orientation
    cross_product = np.cross(v1, v2)

    # Normal Vector
    norm0 = cross_product / np.linalg.norm(cross_product)
    # Transformation to MHD coordinate frame
    norm = [0, 0, 0]
    norm[0] = norm0[1]
    norm[1] = norm0[2]
    norm[2] = norm0[0]

    # Derive the cartesian coordinates of a normalized vector pointing in the direction
    # of the coronal loop's spherical coordinates (midpoint of footpoints)
    midptn_cart = stc(1, theta0, phi0)

    # North Vector
    north0 = [midptn_cart[0].value, midptn_cart[1].value, midptn_cart[2].value]
    # Transformation to MHD coordinate frame
    north = [0, 0, 0]
    north[0] = north0[1]
    north[1] = north0[2]
    north[2] = north0[0]

    # normal vector, north vector
    return norm, north


# Method to load maps from pickle file
def load_maps():
    maps = []

    mapdirs171 = ['maps/0_AIA-STEREOA_171_2012.pkl', 'maps/1_AIA-STEREOA_171_2012.pkl']
    mapdirs195 = ['maps/0_AIA-STEREOA_195_2012.pkl', 'maps/1_AIA-STEREOA_195_2012.pkl']
    mapdirs304 = ['maps/0_AIA-STEREOA_304_2012.pkl', 'maps/1_AIA-STEREOA_304_2012.pkl']

    for name in mapdirs195:
        with open(name, 'rb') as f:
            maps.append(pickle.load(f))
            f.close()

    num_maps = len(maps)

    maps = [maps[0]]

    return maps


img_path = ('/home/saber/CoronalLoopBuilder/examples/testing/downloaded_events/'
            'aia_lev1_193a_2012_07_19t06_40_08_90z_image_lev1.fits')
params_path = 'loop_params/synth_test_AIA2012.pkl'
# params_path = 'loop_params/AIA_2012.pkl'

maps = load_maps()

norm, north = calc_vect(pkl=params_path)
# n, o, v = circle3d(pkl=params_path)
#
# nn = [0, 0, 0]
# nn[0] = n[2]
# nn[1] = n[1]
# nn[2] = -n[0]
#
# no = [0, 0, 0]
# no[0] = o[1]
# no[1] = -o[0]
# no[2] = o[2]

fig = plt.figure()

synth_axs = [synthmap_plot(img_path, fig, normvector=norm, northvector=north, comp=True)]
# synth_axs = [synthmap_plot(img_path, fig, normvector=no, northvector=nn, comp=False)]

coronal_loop1 = CoronalLoopBuilder(fig, synth_axs, maps, pkl=params_path)

plt.show()
plt.close()

coronal_loop1.save_params_to_pickle("synth_test_AIA2012.pkl")
# coronal_loop1.save_to_fig("figs/loop_synth_over.jpg", dpi=300, bbox_inches='tight')



