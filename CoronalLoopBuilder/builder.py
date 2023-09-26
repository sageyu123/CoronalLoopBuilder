import astropy.units as u
from sunpy.coordinates import Heliocentric
from astropy.coordinates import SkyCoord
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import pickle
import warnings
from astropy.time import Time


def circle_3d(x0, y0, z0, r, theta, phi, t):
    """
    Compute the parametric equations for a circle in 3D.

    Parameters:
    - x0, y0, z0: Coordinates of the center of the circle.
    - r: Radius of the circle.
    - theta: Elevation angle of the circle's normal (in radians).
    - phi: Azimuth angle of the circle's normal (in radians).
    - t: Array of parameter values, typically ranging from 0 to 2*pi.

    Returns:
    - x, y, z: Arrays representing the x, y, and z coordinates of the circle in 3D.
    """

    # Normal vector
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # Arbitrary vector (z-axis)
    z = np.array([0, 0, 1])

    # Orthogonal vector in the plane of the circle
    u = np.cross(n, z)
    u /= np.linalg.norm(u)  # Normalize

    # Another orthogonal vector in the plane
    v = np.cross(n, u)
    v /= np.linalg.norm(v)  # Normalize

    # Parametric equations
    x = x0 + r * np.cos(t) * u[0] + r * np.sin(t) * v[0]
    y = y0 + r * np.cos(t) * u[1] + r * np.sin(t) * v[1]
    z = z0 + r * np.cos(t) * u[2] + r * np.sin(t) * v[2]

    return x, y, z


def semi_circle_loop(radius, height, theta0=0 * u.deg, phi0=0 * u.deg, el=90 * u.deg, az=0 * u.deg, samples_num=100):
    '''
    Compute a semicircular loop with both footpoints rooted on the surface of the Sun.

    Parameters:
    - radius: Radius of the semi-circular loop in units compatible with astropy.units (e.g., u.Mm).
    - height: Height of the center of the circle relative to the photosphere in units compatible with astropy.units (e.g., u.Mm).
    - theta0: Heliographic Stonyhurst latitude (theta) of the center of the circle. Default is 0 degrees.
    - phi0: Heliographic Stonyhurst longitude (phi) of the center of the circle. Default is 0 degrees.
    - el: Elevation angle of the circle's normal. It ranges from 0 to 180 degrees. Default is 90 degrees.
    - az: Azimuth angle of the circle's normal. It ranges from 0 to 360 degrees. Default is 0 degrees.
    - samples_num: Number of samples for the parameter t. Default is 1000.


    Returns:
    - SkyCoord object: Represents the coordinates of the semi-circular loop in the heliographic Stonyhurst coordinate system.
    '''

    r_1 = const.R_sun
    # length, height = 500 * u.Mm, 5 * u.Mm

    r0 = r_1 + height
    x0 = u.Quantity(0 * u.cm)
    y0 = u.Quantity(0 * u.cm)
    z0 = r0.to(u.cm)

    theta = el.to(u.rad).value  # np.pi / 2  # Elevation angle
    phi = az.to(u.rad).value  # np.pi / 4  # Azimuth angle
    t = np.linspace(0, 2 * np.pi, int(samples_num))  # Parameter t

    dx, dy, dz = circle_3d(0, 0, 0, radius, theta, phi, t)

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rdiff = r - r_1
    rsort = np.argmin(np.abs(rdiff))
    if rdiff[rsort] + rdiff[rsort + 1] < 0:
        rsort += 1
    r = np.roll(r, -rsort)
    x = np.roll(x, -rsort)
    y = np.roll(y, -rsort)
    z = np.roll(z, -rsort)
    dx = np.roll(dx, -rsort)
    dy = np.roll(dy, -rsort)
    dz = np.roll(dz, -rsort)

    i_r = np.where(r > r_1)
    # r = r[i_r]
    # phi = phi[i_r]
    x = x[i_r]
    y = y[i_r]
    z = z[i_r]
    dx = dx[i_r]
    dy = dy[i_r]
    dz = dz[i_r]

    # Calculate the length of the loop based on the angle between the start and end points.
    # Define the vectors v1 and v2
    v1 = np.array([dx[0].value, dy[0].value, dz[0].value]) * dx[0].unit
    v2 = np.array([dx[-1].value, dy[-1].value, dz[-1].value]) * dx[0].unit
    # Calculate the angle between the vectors (alpha) using the dot product
    cos_alpha = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    alpha = np.arccos(cos_alpha)

    # Use the cross product to determine the orientation
    cross_product = np.cross(v1, v2)
    if cross_product[2] < 0:  # Assuming z is the up direction
        alpha = 2 * np.pi * alpha.unit - alpha

    # Calculate the arc length
    loop_length = alpha.value * radius
    print('Loop length:', loop_length)
    hcc_frame = Heliocentric(observer=SkyCoord(
        lon=phi0, lat=theta0, radius=r_1, frame='heliographic_stonyhurst'))
    return (SkyCoord(x=x, y=y, z=z, frame=hcc_frame).transform_to('heliographic_stonyhurst')), loop_length


class CoronalLoopBuilder:
    """
    Class to build and visualize a coronal loop based on user-defined parameters using sliders.
    """

    def __init__(self, fig, axs, dummy_maps, **kwargs):
        """
        Initialize the CoronalLoopBuilder with given parameters and create the initial visualization.

        Parameters:
        - fig: The figure object.
        - axs: The axes object.
        - dummy_maps: Dummy maps for demonstration.
        - radius: Radius of the semi-circular loop in units compatible with astropy.units (e.g., u.Mm).
        - height: Height of the center of the circle relative to the photosphere in units compatible with astropy.units (e.g., u.Mm).
        - phi0: Heliographic Stonyhurst longitude (phi) of the center of the circle. Default is 0 degrees.
        - theta0: Heliographic Stonyhurst latitude (theta) of the center of the circle. Default is 0 degrees.
        - el: Elevation angle of the circle's normal. It ranges from 0 to 180 degrees. Default is 90 degrees.
        - az: Azimuth angle of the circle's normal. It ranges from 0 to 360 degrees. Default is 0 degrees.
        - samples_num: Number of samples for the parameter t. Default is 1000.
        """
        DEFAULT_RADIUS = 10.0 * u.Mm
        DEFAULT_HEIGHT = 0.0 * u.Mm
        DEFAULT_PHI0 = 0.0 * u.deg
        DEFAULT_THETA0 = 0.0 * u.deg
        DEFAULT_EL = 90.0 * u.deg
        DEFAULT_AZ = 0.0 * u.deg
        DEFAULT_SAMPLES_NUM = 1000
        DEFAULT_loop_color = 'C0'
        DEFAULT_center_color = 'C3'

        self.fig = fig
        self.axs = axs
        self.dummy_maps = dummy_maps

        # Set the loop parameters using the provided values or default values
        self.radius = kwargs.get('radius', DEFAULT_RADIUS)
        self._height = kwargs.get('height', DEFAULT_HEIGHT)
        self.phi0 = kwargs.get('phi0', DEFAULT_PHI0)
        self.theta0 = kwargs.get('theta0', DEFAULT_THETA0)
        self.el = kwargs.get('el', DEFAULT_EL)
        self.az = kwargs.get('az', DEFAULT_AZ)
        self.samples_num = kwargs.get('samples_num', DEFAULT_SAMPLES_NUM)
        self.loop_color = kwargs.get('loop_color', DEFAULT_loop_color)
        self.center_color = kwargs.get('center_color', DEFAULT_center_color)

        # Store the initial values as class attributes
        self.initial_radius = self.radius
        self.initial_height = self.height
        self.initial_phi0 = self.phi0
        self.initial_theta0 = self.theta0
        self.initial_el = self.el
        self.initial_az = self.az
        self.initial_samples_num = self.samples_num

        # self.radius = radius
        # self._height = height
        # self.phi0 = phi0
        # self.theta0 = theta0
        # self.el = el
        # self.az = az
        # self.samples_num = int(samples_num)

        self.loop_length = None
        self.from_textbox = False
        self.ax_button_sliders_toggle = None
        # a flag to check if the class is still initializing
        self.initializing = True
        self.updating_sliders = False
        # When a slider's value is changed, it triggers its on_changed event.
        # If, within the update function, it programmatically sets the value of a slider
        # (e.g., to enforce some constraints), this will again trigger the on_changed event,
        # leading to redundant calls.
        # Add a flag to indicate whether the update is triggered by a user action or programmatically
        self.programmatic_update = False

        self.lines = []
        self.ptns = []
        self.loop_coords = self._compute_loop()
        self.midptn_coords = (
            SkyCoord(lon=self.phi0, lat=self.theta0, radius=const.R_sun, frame='heliographic_stonyhurst'))

        for ax, dummy_map in zip(self.axs, self.dummy_maps):
            line, = ax.plot_coord(self.loop_coords.transform_to(dummy_map.coordinate_frame), color=self.loop_color, lw=2)
            ptn, = ax.plot_coord(self.midptn_coords.transform_to(dummy_map.coordinate_frame), color=self.center_color, marker='o',
                                 ms=3)
            self.lines.append(line)
            self.ptns.append(ptn)
        self._init_sliders()
        self._init_toggle_button()
        # plt.close(self.slider_fig)
        # self.slider_fig = None

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value > self.radius:
            warnings.warn("Height exceeds recommended value! Adjusting to 90% of radius.")
            self._height = self.radius * 0.9
        elif value < -self.radius:
            warnings.warn("Height is too negative! Adjusting to -90% of radius.")
            self._height = -self.radius * 0.9
        else:
            self._height = value

    def _init_sliders(self):
        """
        Initialize sliders for adjusting loop parameters.
        """
        self.updating_sliders = True
        # Create a separate figure for the sliders
        self.slider_fig, axs_sliders = plt.subplots(nrows=7, ncols=2, figsize=(6, 3), width_ratios=[5, 1])
        self.slider_fig.subplots_adjust(left=0.3, wspace=0.1)

        # # Create sliders
        ax_slider_radius, ax_slider_height, ax_slider_phi0, ax_slider_theta0, \
            ax_slider_el, ax_slider_az, ax_slider_samples_num = axs_sliders[:, 0]
        self.slider_radius = Slider(ax_slider_radius, 'Radius [Mm]', 3, 500, valinit=self.radius.to(u.Mm).value)
        self.slider_height = Slider(ax_slider_height, 'Height [Mm]', -300, 300, valinit=self.height.to(u.Mm).value)
        self.slider_phi0 = Slider(ax_slider_phi0, r'HGLN $\Phi$ [deg]', 0, 360, valinit=self.phi0.to(u.deg).value)
        self.slider_theta0 = Slider(ax_slider_theta0, r'HGLT $\Theta$ [deg]', -90, 90,
                                    valinit=self.theta0.to(u.deg).value)
        self.slider_el = Slider(ax_slider_el, 'Elevation [deg]', 0, 180, valinit=self.el.to(u.deg).value)
        self.slider_az = Slider(ax_slider_az, 'Azimuth [deg]', 0, 360, valinit=self.az.to(u.deg).value)
        self.slider_samples_num = Slider(ax_slider_samples_num, 'Samples Num', 10, 2000, valinit=self.samples_num,
                                         valstep=10)
        self.slider_samples_num.valtext.set_text('{:.0f}'.format(self.slider_samples_num.val))  # Display as integer

        self.slider_radius.valtext.set_visible(False)
        self.slider_height.valtext.set_visible(False)
        self.slider_phi0.valtext.set_visible(False)
        self.slider_theta0.valtext.set_visible(False)
        self.slider_el.valtext.set_visible(False)
        self.slider_az.valtext.set_visible(False)
        self.slider_samples_num.valtext.set_visible(False)

        def submit_radius(text):
            self.from_textbox = True
            self.slider_radius.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        def submit_height(text):
            self.from_textbox = True
            self.slider_height.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        def submit_phi0(text):
            self.from_textbox = True
            self.slider_phi0.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        def submit_theta0(text):
            self.from_textbox = True
            self.slider_theta0.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        def submit_el(text):
            self.from_textbox = True
            self.slider_el.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        def submit_az(text):
            self.from_textbox = True
            self.slider_az.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        def submit_samples_num(text):
            self.from_textbox = True
            self.slider_samples_num.set_val(float(text))
            self.from_textbox = False
            self._update(None)

        axbox_radius = axs_sliders[0, 1]
        self.text_box_radius = TextBox(axbox_radius, '')
        self.text_box_radius.on_submit(submit_radius)
        self.text_box_radius.set_val(f"{self.radius.to(u.Mm).value:.1f}")

        axbox_height = axs_sliders[1, 1]
        self.text_box_height = TextBox(axbox_height, '')
        self.text_box_height.on_submit(submit_height)
        self.text_box_height.set_val(f"{self.height.to(u.Mm).value:.1f}")

        axbox_phi0 = axs_sliders[2, 1]
        self.text_box_phi0 = TextBox(axbox_phi0, '')
        self.text_box_phi0.on_submit(submit_phi0)
        self.text_box_phi0.set_val(f"{self.phi0.to(u.deg).value:.2f}")

        axbox_theta0 = axs_sliders[3, 1]
        self.text_box_theta0 = TextBox(axbox_theta0, '')
        self.text_box_theta0.on_submit(submit_theta0)
        self.text_box_theta0.set_val(f"{self.theta0.to(u.deg).value:.2f}")

        axbox_el = axs_sliders[4, 1]
        self.text_box_el = TextBox(axbox_el, '')
        self.text_box_el.on_submit(submit_el)
        self.text_box_el.set_val(f"{self.el.to(u.deg).value:.2f}")

        axbox_az = axs_sliders[5, 1]
        self.text_box_az = TextBox(axbox_az, '')
        self.text_box_az.on_submit(submit_az)
        self.text_box_az.set_val(f"{self.az.to(u.deg).value:.2f}")

        axbox_samples_num = axs_sliders[6, 1]
        self.text_box_samples_num = TextBox(axbox_samples_num, '')
        self.text_box_samples_num.on_submit(submit_samples_num)
        self.text_box_samples_num.set_val(f"{int(self.samples_num):.0f}")

        # Attach update function to sliders
        self.slider_radius.on_changed(self._update)
        self.slider_height.on_changed(self._update)
        self.slider_theta0.on_changed(self._update)
        self.slider_phi0.on_changed(self._update)
        self.slider_el.on_changed(self._update)
        self.slider_az.on_changed(self._update)
        self.slider_samples_num.on_changed(self._update_samples_num)

        # Create a button to print the values
        ax_button_print = self.slider_fig.add_axes([0.7, 0.02, 0.08, 0.07])
        self.button_print_values = plt.Button(ax_button_print, 'Print')
        self.button_print_values.on_clicked(self._print_values)

        # Add "Reset" button next to "Print Values" button
        ax_reset = plt.axes([0.6, 0.02, 0.08, 0.07])
        self.button_reset = plt.Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self._reset_sliders)

        # turn the flag off when the class is still initialized
        self.initializing = False
        self.updating_sliders = False

    def _init_toggle_button(self):
        """
        Initialize a button to toggle the visibility of the sliders.
        """
        # Add a button to toggle the sliders figure
        if not hasattr(self, 'button_toggle_sliders'):
            self.ax_button_sliders_toggle = self.fig.add_axes([0.8, 0.9, 0.1, 0.04])
            self.button_toggle_sliders = plt.Button(self.ax_button_sliders_toggle, 'Settings')
            self.button_toggle_sliders.on_clicked(self._toggle_sliders)

    def _toggle_sliders(self, event):
        """
        Toggle the visibility of the sliders.
        """

        if self.slider_fig is not None and plt.fignum_exists(self.slider_fig.number):
            plt.close(self.slider_fig)
            self.slider_fig = None
            self.initializing = True
        else:
            self._init_sliders()
            self.initializing = True
            # Set the sliders to the current values
            self._init_sliders_value(None)
            self._init_toggle_button()

    def _init_sliders_value(self, event):
        self.initializing = True
        # Set the sliders to the current values
        self.slider_radius.set_val(self.radius.to(u.Mm).value)
        self.slider_height.set_val(self.height.to(u.Mm).value)
        self.slider_phi0.set_val(self.phi0.to(u.deg).value)
        self.slider_theta0.set_val(self.theta0.to(u.deg).value)
        self.slider_el.set_val(self.el.to(u.deg).value)
        self.slider_az.set_val(self.az.to(u.deg).value)
        self.initializing = False
        self.slider_samples_num.set_val(int(self.samples_num))

    def _update_samples_num(self, val):
        """
        Update the number of samples and ensure the displayed value is an integer.
        """

        # Ensure the displayed value is an integer
        self.samples_num = int(self.slider_samples_num.val)
        self.slider_samples_num.valtext.set_text(f"{int(self.slider_samples_num.val)}")
        self._update(val)

    def _compute_loop(self):
        """
        Compute the coordinates of the coronal loop based on the current slider values.
        """
        loop, self.loop_length = semi_circle_loop(self.radius, self.height, self.theta0, self.phi0, self.el, self.az,
                                                  int(self.samples_num))
        return loop

    def _print_values(self, event):
        print(f"radius = {self.radius.value:.1f} * u.{self.radius.unit}, "
              f"height = {self.height.value:.1f} * u.{self.height.unit}, "
              f"phi0 = {self.phi0.value:.2f} * u.{self.phi0.unit}, "
              f"theta0 = {self.theta0.value:.2f} * u.{self.theta0.unit}, "
              f"el = {self.el.value:.2f} * u.{self.el.unit}, "
              f"az = {self.az.value:.2f} * u.{self.az.unit}, "
              f"samples_num = {int(self.samples_num)}")

    def _reset_sliders(self, event):
        # Reset the sliders to their initial values
        self.initializing = True
        self.slider_radius.set_val(self.initial_radius.to(u.Mm).value)
        self.slider_height.set_val(self.initial_height.to(u.Mm).value)
        self.slider_phi0.set_val(self.initial_phi0.to(u.deg).value)
        self.slider_theta0.set_val(self.initial_theta0.to(u.deg).value)
        self.slider_el.set_val(self.initial_el.to(u.deg).value)
        self.slider_az.set_val(self.initial_az.to(u.deg).value)
        self.initializing = False
        self.slider_samples_num.set_val(int(self.initial_samples_num))

        # Update the loop with the initial values
        self._update(None)

        print('Restore to initial values')
        self._print_values(None)

    def _update(self, val):
        """
        Update the visualization based on the current slider values.
        """

        if self.programmatic_update or self.initializing or self.updating_sliders or self.from_textbox:
            return

        self.programmatic_update = True
        self.radius = u.Quantity(self.slider_radius.val, u.Mm)
        self.height = u.Quantity(self.slider_height.val, u.Mm)
        self.phi0 = u.Quantity(self.slider_phi0.val, u.deg)
        self.theta0 = u.Quantity(self.slider_theta0.val, u.deg)
        self.el = u.Quantity(self.slider_el.val, u.deg)
        self.az = u.Quantity(self.slider_az.val, u.deg)

        # Ensure height does not exceed radius
        if self.height > self.radius:
            self.height = self.radius * 0.9
            self.slider_height.set_val(self.height.to(u.Mm).value)  # Update the slider value to reflect the change
        if self.height < -self.radius:
            self.height = -self.radius * 0.9
            self.slider_height.set_val(self.height.to(u.Mm).value)  # Update the slider value to reflect the change

        # newheight = u.Quantity(self.slider_height.val, u.Mm)
        # if newheight > self.radius:
        #     self.slider_height.set_val(self.height.to(u.Mm).value)
        # elif newheight < -self.radius:
        #     self.slider_height.set_val(self.height.to(u.Mm).value)
        # else:
        #     self.height = newheight

        self.loop_coords = self._compute_loop()
        self.midptn_coords = (
            SkyCoord(lon=self.phi0, lat=self.theta0, radius=const.R_sun, frame='heliographic_stonyhurst'))

        # Update the lines
        for ptn, line, dummy_map in zip(self.ptns, self.lines, self.dummy_maps):
            midptn_coords = self.midptn_coords.transform_to(dummy_map.coordinate_frame)
            ptn.set_xdata(midptn_coords.spherical.lon.deg)
            ptn.set_ydata(midptn_coords.spherical.lat.deg)
            loop_coords = self.loop_coords.transform_to(dummy_map.coordinate_frame)
            line.set_xdata(loop_coords.spherical.lon.deg)
            line.set_ydata(loop_coords.spherical.lat.deg)

        for ax in self.axs:
            ax.figure.canvas.draw_idle()

        # Update text box values
        if hasattr(self, 'text_box_radius'):
            self.text_box_radius.set_val(f"{self.radius.to(u.Mm).value:.1f}")
        if hasattr(self, 'text_box_height'):
            self.text_box_height.set_val(f"{self.height.to(u.Mm).value:.1f}")
        if hasattr(self, 'text_box_phi0'):
            self.text_box_phi0.set_val(f"{self.phi0.to(u.deg).value:.2f}")
        if hasattr(self, 'text_box_theta0'):
            self.text_box_theta0.set_val(f"{self.theta0.to(u.deg).value:.2f}")
        if hasattr(self, 'text_box_el'):
            self.text_box_el.set_val(f"{self.el.to(u.deg).value:.2f}")
        if hasattr(self, 'text_box_az'):
            self.text_box_az.set_val(f"{self.az.to(u.deg).value:.2f}")
        if hasattr(self, 'text_box_samples_num'):
            self.text_box_samples_num.set_val(f"{int(self.samples_num)}")

        self.programmatic_update = False

    def save_loop_data_to_pickle(self, filename="coronal_loop.pkl"):
        """
        Save the loop data to a pickle file.

        Parameters:
        - filename (str): The name of the pickle file to save to. Defaults to "coronal_loop.pkl".
        """
        data_to_save = {
            "loop": self.loop_coords,
            "length": self.loop_length
        }

        with open(filename, 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Loop coords data saved to {filename}!")

    def save_to_fig(self, figname, **kwargs):
        # Hide the "Settings" button
        if hasattr(self, 'button_toggle_sliders'):
            self.button_toggle_sliders.ax.set_visible(False)
            # self.toggle_sliders(1)
            self.fig.canvas.draw_idle()

        # Save the figure with additional kwargs
        self.fig.savefig(figname, **kwargs)

        # Restore the "Settings" button
        if hasattr(self, 'button_toggle_sliders'):
            # self.toggle_sliders(1)
            self.button_toggle_sliders.ax.set_visible(True)
            self.fig.canvas.draw_idle()
        print(f"Loop fig saved to {figname}!")

    def set_loop(self, **kwargs):
        """
        Update the loop using the provided parameters.

        Parameters:
        - radius: Radius of the semi-circular loop in units compatible with astropy.units (e.g., u.Mm).
        - height: Height of the center of the circle relative to the photosphere in units compatible with astropy.units (e.g., u.Mm).
        - phi0: Heliographic Stonyhurst longitude (phi) of the center of the circle. Default is 0 degrees.
        - theta0: Heliographic Stonyhurst latitude (theta) of the center of the circle. Default is 0 degrees.
        - el: Elevation angle of the circle's normal. It ranges from 0 to 180 degrees. Default is 90 degrees.
        - az: Azimuth angle of the circle's normal. It ranges from 0 to 360 degrees. Default is 0 degrees.
        - samples_num: Number of samples for the parameter t. Default is 1000.
        """
        # Update the loop parameters with the provided values
        if 'radius' in kwargs:
            self.radius = kwargs['radius']
        if 'height' in kwargs:
            self.height = kwargs['height']
        if 'phi0' in kwargs:
            self.phi0 = kwargs['phi0']
        if 'theta0' in kwargs:
            self.theta0 = kwargs['theta0']
        if 'el' in kwargs:
            self.el = kwargs['el']
        if 'az' in kwargs:
            self.az = kwargs['az']
        if 'samples_num' in kwargs:
            self.samples_num = kwargs['samples_num']

        if self.slider_fig is not None and plt.fignum_exists(self.slider_fig.number):
            self.initializing = True
            # Set the sliders to the current values
            self._init_sliders_value(None)
            # Now, recompute the loop with the updated parameters

        self._update(None)

        # Redraw the figure to reflect the changes
        self.fig.canvas.draw()
