import numpy as np
from scipy.spatial import Voronoi, KDTree, Delaunay
import translate 

# Brillouin Zone object
class BZ():
    def __init__(
        self, rec_latt:np.ndarray
    )->None:
        self.rec_latt = rec_latt
        _3xgrid, _ = translate.TranslatePoints(np.zeros((1,3)), np.zeros(1), self.rec_latt)
        self._3xgrid = _3xgrid
        self.vertices = self._BZpts()
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # Method used to check if nearest points all occur in same plane
    def _CheckPts(
        self, pts:np.ndarray, verts:np.ndarray
    )->bool:
        # get cartesian coordinates
        verts = np.take(verts, pts, axis=0)
        # last point is the furthest, check if it is in same plane as other points
        pt = verts[-1,:]
        check = False
        # truncate floats to avoid floating point errors when checking for equivalence
        pt = [np.sign(n)*float(str(np.abs(n))[:8]) for n in pt]
        for i, vert in enumerate(verts):
            verts[i,:] = [np.sign(n)*float(str(np.abs(n))[:8]) for n in vert]
        # check if points are in same plane
        if pt[0] == verts[0,:][0] and pt[0] == verts[1,:][0] and pt[0] == verts[2,:][0]:
            check = True 
        if pt[1] == verts[0,:][1] and pt[1] == verts[1,:][1] and pt[1] == verts[2,:][1]:
            check = True
        if pt[2] == verts[0,:][2] and pt[2] == verts[1,:][2] and pt[2] == verts[2,:][2]:
            check = True
        return check
    #-----------------------------------------------------------------------------------------------------------------#
    def _BZpts(
        self
    )->np.ndarray:
        # Voronoi tessallate the lattice points
        vor_tessellation = Voronoi(self._3xgrid)
        # find Voronoi region that encapsulates Brillouin zone
        vor_cell = -1
        for i, region in enumerate(vor_tessellation.regions):
            if -1 not in region and region != []:
                vor_cell = i
                break
        if vor_cell < 0:
            raise ValueError(
                'Unable to construct Brillouin Zone from reciprocal lattice, check lattice parameters.'
            )
        # get vertices of BZ Voronoi region
        vor_cell_verts = vor_tessellation.regions[vor_cell]
        vor_verts = np.take(vor_tessellation.vertices, vor_cell_verts, axis=0)
        # find which vertices are closest to each other
        point_cloud = KDTree(vor_verts)
        vert_inds = []
        for i, vert in enumerate(vor_verts):
            _, nearest_pts = point_cloud.query(vert, k=4)
            c = 4
            # iterate check until points are no longer in same plane
            while self._CheckPts(nearest_pts, vor_verts):
                c += 1
                _, nearest_pts = point_cloud.query(vert, k=c)
                len_arr = c - 4
                arr = -1*np.arange(len_arr) - 2
                nearest_pts = np.delete(nearest_pts, arr, axis=0)
            # current index is a repeated entry every other point since this is how PyVista interprets lines
            nearest_pts = np.insert(nearest_pts, [2,4], [i,i])
            vert_inds.append(nearest_pts)
        # convert nested list to flat list
        vert_inds = [int(ind) for verts in vert_inds for ind in verts]
        # get cartesian coordinates
        vor_verts = np.take(vor_verts, vert_inds, axis=0)
        return vor_verts
    #-----------------------------------------------------------------------------------------------------------------#
    # method for determining shifts required to translate points outside BZ back into BZ
    def GetShifts(
        self, points:np.ndarray, cart:bool=True
    )->np.ndarray:
        '''
        Method for calculating necessary reciprocal lattice shift to move point into Brillouin Zone.

        Parameters
        ----------
        points : np.ndarray
            List of points to calculate shifts for.
        cart : bool
            Convert points from reduced to Cartesian coordinates.
            Default will apply conversion (True).
        '''
        # make points mulitdimensional if not already
        if len(points.shape) == 1:
            points = points.reshape((1,points.shape[0]))
        # convert to cartesian
        if cart:
            points = np.matmul(points, self.rec_latt)
        # find points outside of BZ
        outside_pts = Delaunay(self.vertices).find_simplex(points)
        # calculate shifts to move outside points back into BZ
        shifts = np.zeros((outside_pts[outside_pts < 0].shape[0],3))
        rec_grid = KDTree(self._3xgrid)
        shift_grid, _ = translate.TranslatePoints(np.zeros((1,3)), np.zeros(1), np.identity(3))
        for i, pt in enumerate(points[outside_pts < 0]):
            _, closest_pt = rec_grid.query(pt, k=1)
            shifts[i,:] = shift_grid[closest_pt, :]
        all_shifts = np.zeros((points.shape[0],3))
        all_shifts[outside_pts < 0] = shifts
        return all_shifts.astype(int)