import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors

# some animation settings
interval = 50
save_gif = False
gif_fps = 30
repeat_delay_frames = 20   # for gif output

def animate_oas_solutions(n_data, mesh_his, mesh_undeform_his, sec_forces_his, fem_sol=None, symmetry=True, save_static_index=0):
    """
    Plots OAS mesh, aero forces, and structural stress distributions
    Creates animation for dynamics or optimization iterations. The first index of the inputs

    Inputs:
        n_data : int
            Number of data points. This can be the number of optimization iterations, or the number of time steps for dynamics, or just 1 to plot a single wing.
        mesh_his : ndarray, (n_data, nx, ny, 3)
            Deformed mesh history.
        mesh_undeform_his : ndarray, (n_data, nx, ny, 3)
            Undeformed mesh history
        sec_forces_his : ndarray, (n_data, nx-1, ny-1, 3)
            Sectional force history
        stress_margin_his : ndarray, (n_data, ny-1, 2)
            Stress margin history (computed based on Von Mises stress and yield stress)
        fem_sol : dict
            If None, just plot the aero solution.
            Dictionary containing structural tube parameters. Keys are
            'radius_his'     : tube radius history, (n_data, ny-1,)
            'thickness'  : tube thickness history, (n_data, ny-1,)
            'stress_margin_his' : stress margin history, (n_data, ny-1, 2)
            'fem_origin' : chordwise location of the tube, float
        symmetry : bool
            True for symmetry solutions, False if not
        save_static_index : int
            Save static figure for this time index. 0 for initial data, -1 for last data.
    """

    # get mesh shape
    _, nx, ny, _ = mesh_his.shape

    if symmetry:
        # --- create right-wing mesh for symmetry case ---
        # flip the mesh for the right-wing
        mirror_mesh_his = mesh_his.copy()
        mirror_mesh_his[:, :, :, 1] *= -1.0
        mirror_mesh_his = mirror_mesh_his[:, :, ::-1, :]
        ### mesh_full = np.concatenate((mesh_his, mirror_mesh), axis=2)

        # also flip the undeformed mesh because we plot the undeformed mesh on the right
        mirror_mesh_undeform_his = mesh_undeform_his.copy()
        mirror_mesh_undeform_his[:, :, :, 1] *= -1.0
        mirror_mesh_undeform_his = mirror_mesh_undeform_his[:, ::-1, :]
    else:
        # --- split the full mesh into left-wing and right-wing ---
        ny = mesh_his.shape[2]
        if ny % 2 != 1:
            raise RuntimeError('OAS mesh ny is not odd')
        ny_center = int((ny - 1) / 2)

        # right-wing mesh
        mirror_mesh_his = mesh_his[:, :, ny_center:, :]
        mirror_mesh_undeform_his = mesh_undeform_his[:, :, ny_center:, :]

        # left-wing mesh
        mesh_his = mesh_his[:, :, :ny_center + 1, :]
        mesh_undeform_his = mesh_undeform_his[:, :, :ny_center + 1, :]

        # sectional force for left-wing
        sec_forces_his = sec_forces_his[:, :, :ny_center, :]

        # FEM data for left-wing
        if fem_sol is not None:
            fem_sol['radius_his'] = fem_sol['radius_his'][:, :ny_center]
            fem_sol['thickness_his'] = fem_sol['thickness_his'][:, :ny_center]
            fem_sol['stress_margin_his'] = fem_sol['stress_margin_his'][:, :ny_center, :]
        print('WARNING: Only the left-wing aerostructural solution will be plotted even if the wing is symmetric')

        # mesh size for symmetric wing
        _, nx, ny, _ = mesh_his.shape
    # END IF

    # color mormalization for structural stress. Put minus to reverse the colormap
    if fem_sol is not None:
        stress_min = np.min(-fem_sol['stress_margin_his'])   # ~ -1 (100% stress margin)
        stress_max = np.max(-fem_sol['stress_margin_his'])   # ~ 0 (0% stress margin)
        norm_stress = colors.Normalize(vmin=stress_min, vmax=stress_max)

    # color normalization for aero force
    vminFz = np.amin(sec_forces_his[:, :, :, 2])   # max of Fz
    vmaxFz = np.amax(sec_forces_his[:, :, :, 2])
    norm_aero_force = colors.Normalize(vmin=vminFz, vmax=vmaxFz)

    # initialize figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    
    def init():
        pass
        
    def animate(t_idx):
        if t_idx >= n_data:
            t_idx = n_data - 1

        ax.clear()

        # --- plot the deformed mesh on the left side as wireframe ---
        mesh_x = mesh_his[t_idx, :, :, 0]
        mesh_y = mesh_his[t_idx, :, :, 1]
        mesh_z = mesh_his[t_idx, :, :, 2]
        ax.plot_wireframe(mesh_x, mesh_y, mesh_z, rstride=1, cstride=1, color="black", linewidths=0.2)

        # --- plot tube structure on the left side with stress-margin colors ---
        if fem_sol is not None:
            # Get the array of radii and thickness values for the FEM system
            r0 = fem_sol['radius_his'][t_idx, :]
            t0 = fem_sol['thickness_his'][t_idx, :]
            stress_margin = fem_sol['stress_margin_his'][t_idx, :, :]   # (ny - 1, 2). This needs to be >=0. 1 for 0 stress, 0 for yield stress
            stress_margin = np.min(stress_margin, axis=1)

            # Create a normalized array of values for the colormap
            colors_stress = -stress_margin   # put minus here to reverse the colormap. We'll get yellow for 0% margin (higher stress), blue for 100% margin

            # Set the number of rectangular patches on the cylinder
            num_circ = 12
            fem_origin = fem_sol['fem_origin']

            # Create an array of angles around a circle
            p = np.linspace(0, 2 * np.pi, num_circ)

            mesh0 = mesh_his[t_idx, :, :, :]

            # Loop through each element in the FEM system
            for i, thick in enumerate(t0):
                # Get the radii describing the circles at each nodal point
                r = np.array((r0[i], r0[i]))
                R, P = np.meshgrid(r, p)

                # Get the X and Z coordinates for all points around the circle
                X, Z = R * np.cos(P), R * np.sin(P)

                # Get the chord and center location for the FEM system
                chords = mesh0[-1, :, 0] - mesh0[0, :, 0]
                comp = fem_origin * chords + mesh0[0, :, 0]

                # Add the location of the element centers to the circle coordinates
                X[:, 0] += comp[i]
                X[:, 1] += comp[i + 1]
                Z[:, 0] += fem_origin * (mesh0[-1, i, 2] - mesh0[0, i, 2]) + mesh0[0, i, 2]
                Z[:, 1] += fem_origin * (mesh0[-1, i + 1, 2] - mesh0[0, i + 1, 2]) + mesh0[0, i + 1, 2]

                # Get the spanwise locations of the spar points
                Y = np.empty(X.shape)
                Y[:] = np.linspace(mesh0[0, i, 1], mesh0[0, i + 1, 1], 2)

                # Set the colors of the rectangular surfaces
                col = np.zeros(X.shape)
                col[:] = colors_stress[i]

                # Plot the rectangular surfaces for each individual FEM element
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.viridis(norm_stress(col)), linewidth=0)
            # END FOR
        # END IF (plotting FEM tube)

        # --- plot panel force distribution on the right side ---
        # panel force in z direction
        Fz = sec_forces_his[t_idx, :, :, 2]   # (nx - 1, ny - 1)

        # project panel force into nodes.
        nodal_force = np.zeros((nx, ny))
        nodal_force[0:-1, 0:-1] += 0.75 / 2 * Fz[:, :]   # front-left
        nodal_force[0:-1, 1:] += 0.75 / 2 * Fz[:, :]   # front-right
        nodal_force[1:, 0:-1] += 0.25 / 2 * Fz[:, :]   # rear-left
        nodal_force[1:, 1:] += 0.25 / 2 * Fz[:, :]   # rear-right
        # override edges
        nodal_force[0, 0:-1] = 0.5 * Fz[0, :]   # leading edge
        nodal_force[0, 1:] += 0.5 * Fz[0, :]
        nodal_force[-1, 0:-1] = 0.5 * Fz[-1, :]   # trailing edge
        nodal_force[-1, 1:] += 0.5 * Fz[-1, :]
        nodal_force[0:-1, 0] = 0.5 * Fz[:, 0]   # wing tip
        nodal_force[1:, 0] += 0.5 * Fz[:, 0]
        nodal_force[0:-1, -1] = 0.5 * Fz[:, -1]   # wing root
        nodal_force[1:, -1] += 0.5 * Fz[:, -1]
        # corners
        nodal_force[0, 0] = Fz[0, 0]
        nodal_force[0, -1] = Fz[0, -1]
        nodal_force[-1, 0] = Fz[-1, 0]
        nodal_force[-1, -1] = Fz[-1, -1]
        # flip y for right wing
        nodal_force = nodal_force[:, ::-1]

        # plot force distribution on the deformed mesh
        mesh_x = mirror_mesh_his[t_idx, :, :, 0]
        mesh_y = mirror_mesh_his[t_idx, :, :, 1]
        mesh_z = mirror_mesh_his[t_idx, :, :, 2]
        ax.plot_surface(mesh_x, mesh_y, mesh_z, facecolors=cm.viridis(norm_aero_force(nodal_force)), cmap=cm.viridis)

        # also plot the undeformed mesh for reference
        ax.plot_wireframe(mirror_mesh_undeform_his[t_idx, :, :, 0], mirror_mesh_undeform_his[t_idx, :, :, 1], mirror_mesh_undeform_his[t_idx, :, :, 2], rstride=1, cstride=1, color="gray", linewidths=0.2)
        
        ax.set_axis_off()
        ax.set_xlim([0, 0.3])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([-0.2, 0.])
        ax.set_aspect('equal')
        ax.view_init(azim=-130, elev=50)

    # save static figure
    animate(save_static_index)
    plt.savefig('wing_index' + str(save_static_index) + '.pdf', bbox_inches='tight')

    # plot animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_data + repeat_delay_frames, interval=interval, repeat=True, repeat_delay=1000)
    if save_gif:
        anim.save("animation_wing_sol.gif", writer=animation.PillowWriter(fps=gif_fps))

    plt.show()