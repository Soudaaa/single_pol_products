import pyart
import numpy as np
import copy
from scipy.interpolate import interp1d, interp2d
from numba import njit

def z2r(ref = None):
    """
    Converts the horizontal reflectivity factor from logarithmic to linear units.
    """
    return 10**(ref/10)

def calc_VIL(radar = None, zh_name = 'corrected_reflectivity', vil_name = 'VIL'):
    """
    Evaluates the Vertically Integrated Liquid (VIL) as in Amburn and Wolf (1997).
    In order for this function to work, all azimuths and elevations (if not constant as with some brazilian radars)
    must be sorted following an increasing order.
    Number of azimuths between sweeps also must have the same shape. If not, it's recommended to use the sort_radar function
    which interpolates along azimuths to maintain the same azimutal shape.
    
    References
    __________
    
    Amburn, S. A., &; Wolf, P. L. (1997). VIL Density as a Hail Indicator. <i>Weather and Forecasting, 12(3), 473–478.
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
    zh_name: str
        Name of the horizontal reflectivity factor field. Default is corrected_reflectivity
    threshold: int of float
        minimum reflectivity threshold need to seek for the echo tops
    et_name: str
        name used for the Echo Tops field. Default is ET.
        
    Returns
    __________
    
    Radar object with the ET field included (m)
    
    """
    const = 3.44e-6
    # get all the sweeps' azimuths along and store them on a list for later. Also get the number of rays or azimuths of each sweep.
    azi = [radar.azimuth['data'][x] for x in radar.iter_slice()]
    n_rays = azi[0].shape[0]
    
    # get the elevation angles of each sweep. Sort them from the lowest to highest
    ele = radar.fixed_angle['data']
    sort_idx = np.argsort(ele)
    
    # get the number of sweeps
    n_sweeps = radar.nsweeps
    
    # get the radar range at the lowest elevation and the number of gates along each ray
    rg = radar.range['data']
    n_bins = radar.ngates
    
    # create empty matrixes to stack the reflectivity and x, y, z coordinates of every gate of each gate for every sweep.
    # Also create an empty matrix to store the heights between gates' centers and the eventual VIL
    ref_stacked = np.zeros((n_sweeps, n_rays, n_bins))
    x_stacked = np.zeros_like(ref_stacked)
    y_stacked = np.zeros_like(ref_stacked)
    z_stacked = np.zeros_like(ref_stacked)
    dz_stacked = np.zeros_like(ref_stacked)
    VIL = np.zeros_like(ref_stacked[0, :, :])
    
    # iterate over each sweep and stack the reflectivity, x, y, z coordinates
    for i in sort_idx:
        ref_stacked[i, :, :] = radar.get_field(i, zh_name, copy = True)
        x, y, z = radar.get_gate_x_y_z(i)
        x_stacked[i, :, :] = x
        y_stacked[i, :, :] = y
        z_stacked[i, :, :] = z + radar.altitude['data']
    
    # convert the reflectivity factor to linear units
    ref = np.array(z2r(ref_stacked))
    
    # get the range of each gate
    r = np.sqrt(x_stacked**2 + y_stacked**2)
    
    # compute the dz of each gate between sweeps
    for i in range(n_sweeps):
        if i < n_sweeps - 1:
            dz_stacked[i, :, :] = z_stacked[i + 1, :, :] - z_stacked[i, :, :]
        if i == n_sweeps - 1:
            dz_stacked[i, :, :] = z_stacked[i, :, :] - z_stacked[i - 1, :, :]
    
    # mask gates where the reflectivity is below 0 dBZ
    valid = (ref_stacked > 1)
    
    # compute VIL for every gate
    VIL_elements = const*(ref**(4/7))*dz_stacked
    
    # loop over every ray seeking for valid gates
    for az_idx in range(n_rays):
        slice_valid = valid[:, az_idx, :]
        if not np.any(slice_valid):
            continue
        # keep the valid gates
        slice_VIL_elements = VIL_elements[:, az_idx, :]
        # loop over every range gate
        for rg_idx in range(n_bins):
            VIL_temp = 0
            sfc_rg = rg[rg_idx]
            #loop over every sweep 
            for el_idx in range(n_sweeps):
                if not slice_valid[el_idx, rg_idx]:
                    continue
                if slice_VIL_elements[el_idx, rg_idx] == 0:
                    continue
                # if the first sweep, keep the VIL value
                if el_idx == 0:
                    VIL_temp = slice_VIL_elements[el_idx, rg_idx]
                # if not, keep iterating along every ray of every sweep
                else:
                    # this is done in order to fit higher tilt rays with the nearest ray of the first sweep
                    ppi_ray_rg = r[el_idx, az_idx, :]
                    closest_idx = np.argmin(np.abs(ppi_ray_rg - sfc_rg))
                    # keep looking for valid VIL values at higher sweeps
                    VIL_temp += slice_VIL_elements[el_idx, closest_idx]
            # once the iteration is finished, vertically sum the VIL values of each gate and store them in the lowest sweep
            if VIL_temp > 0:
                VIL[az_idx, rg_idx] = np.nansum(VIL_temp)
                
    # mask gates where the reflectivity is below 0 dBZ          
    VIL = np.ma.masked_where(ref_stacked[0] <= 1, VIL)
    
    # add 0 values to higher sweeps
    VIL_field = np.zeros_like(radar.fields[zh_name]['data'])
    VIL_field[radar.get_slice(sort_idx[0])] = VIL
    
    # add the VIL field to the radar object
    radar.add_field(vil_name, dic = {'units':'kg m$^{-2}$',
                                  'data':VIL_field, 'standard_name':'vertically_integrated_liquid',
                                  'long_name':'Vertically Integrated Liquid', 
                                  'coordinates':'elevation azimuth range'}, replace_existing=True)
    
    return radar

def calc_ET(radar = None, zh_name = 'corrected_reflectivity', threshold = 18., et_name = 'ET'):
    """
    Retrieves the echo tops based on a minimum horizontal reflectivity threshold.
    In order for this function to work, all azimuths and elevations (if not constant, as with some brazilian radar)
    must be sorted following an increasing order.
    Number of azimuths between sweeps also must have the same shape. If not, it's recommended to use the sort_radar function
    which interpolates along azimuths to maintain the same azimutal shape.
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
    zh_name: str
        Name of the horizontal reflectivity field factor. Default is corrected_reflectivity
    threshold: int of float
        minimum reflectivity threshold need to seek for the echo tops
    et_name: str
        name used for the Echo Tops field. Default is ET.
        
    Returns
    __________
    
    Radar object with the ET field included (m)
    
    """
    
    # get all the sweeps' azimuths along and store them on a list for later. Also get the number of rays or azimuths of each sweep.
    azi = [radar.azimuth['data'][x] for x in radar.iter_slice()]
    n_rays = azi[0].shape[0]
    
    # get the elevation angles of each sweep. Sort them from the lowest to highest
    ele = radar.fixed_angle['data']
    sort_idx = np.argsort(ele)
    
    # get the number of sweeps
    n_sweeps = radar.nsweeps
    
    # get the radar range at the lowest elevation and the number of gates along each ray
    rg = radar.range['data']
    n_bins = radar.ngates

    # create empty matrixes to stack the reflectivity, x, y and z coordinates of each gate for every sweep. Also create an empty matrix
    # to store the heights between gates and the eventual echo tops
    ref_stacked = np.zeros((n_sweeps, n_rays, n_bins))
    x_stacked = np.zeros_like(ref_stacked)
    y_stacked = np.zeros_like(ref_stacked)
    z_stacked = np.zeros_like(ref_stacked)
    ET = np.zeros((n_rays, n_bins))
    
    # iterate over each sweep and stack the reflectivity, x, y, and z coordinates
    for i in sort_idx:
        ref_stacked[i, :, :] = radar.get_field(i, zh_name, copy = True)
        x, y, z = radar.get_gate_x_y_z(i)
        x_stacked[i, :, :] = x
        y_stacked[i, :, :] = y
        z_stacked[i, :, :] = z
    
    # get the range of each gate
    r = np.sqrt(x_stacked**2 + y_stacked**2)
    
    # create a mask to remove gates below the desired threshold
    valid = (ref_stacked >= threshold)
    
    # loop over every ray seeking for valid gates
    for az_idx in range(n_rays):
        slice_valid = valid[:, az_idx, :]
        if not np.any(slice_valid):
            continue
        # keep the valid gates
        slice_ET_elements = z_stacked[:, az_idx, :]
        # loop over every range gate
        for rg_idx in range(n_bins):
            ET_temp = 0
            sfc_rg = rg[rg_idx]
            #loop over every sweep 
            for el_idx in range(n_sweeps):
                if not slice_valid[el_idx, rg_idx]:
                    continue
                if slice_ET_elements[el_idx, rg_idx] == 0:
                    continue
                # if the first sweep, keep the gate heights
                if el_idx == 0:
                    ET_temp = slice_ET_elements[el_idx, rg_idx]
                # if not, keep iterating along every ray of every sweep
                else:
                    # this is done in order to fit higher tilt rays with the nearest ray of the first sweep
                    ppi_ray_rg = r[el_idx, az_idx, :]
                    closest_idx = np.argmin(np.abs(ppi_ray_rg - sfc_rg))
                    # once the gate heights of each sweep are stored, look for the highest ones to get the echo tops
                    ET_temp = np.nanmax(slice_ET_elements[el_idx, closest_idx])
            # once the iteration is finished, store them in the lowest sweep
            if ET_temp > 0:
                ET[az_idx, rg_idx] = ET_temp
    
    # mask gates where the reflectivity is below the chosen threshold
    ET = np.ma.masked_where(ref_stacked[0] <= threshold, ET)
    
    # add 0 values to higher sweeps
    ET_field = np.zeros_like(radar.fields[zh_name]['data'])
    ET_field[radar.get_slice(sort_idx[0])] = ET
    
    # add the Echo Tops field to the radar
    radar.add_field(et_name, dic = {'units':'m',
                                 'data':ET_field, 'standard_name':f'{threshold}_dBZ_echo_tops',
                                 'long_name':f'{threshold} dBZ echo tops', 
                                 'coordinates':'elevation azimuth range'}, replace_existing=True)
    
    return radar
    
def calc_VILD(radar = None, VIL_name = 'VIL', ET_name = 'ET'):
    """
    Evaluates the Vertically Integrated Liquid Density (VILD) using VIL and 18 dBZ ETs as in Murillo and Homeyer (2019).
    In order for this function to work, all azimuths and elevations (if not constant, as with some brazilian radars)
    must be sorted following an increasing order.
    Number of azimuths between sweeps also must have the same shape. If not, it's recommended to use the sort_radar function
    which interpolates along azimuths to maintain the same azimutal shape.
    
    References
    __________
    
    Murillo, E. M., & Homeyer, C. R. (2019). Severe hail fall and hailstorm detection using remote sensing observations. Journal of Applied Meteorology and Climatology, 58(5), 947–970. https://doi.org/10.1175/JAMC-D-18-0247.1
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
    VIL: str
        Name of the VIL field. Default is VIL
    ET: str
        minimum reflectivity threshold need to seek for the echo tops
    et_name: str
        name used for the Echo Tops field. Default is ET.
        
    Returns
    __________
    
    Radar object with the ET field included (m)
    
    """
    
    VILD_field = radar.fields[VIL_name]['data'] * 1000. / radar.fields[ET_name]['data']
    radar.add_field('VILD', dic = {'units':'g m$^{-3}$',
                                  'data':VILD_field, 'standard_name':'vertically_integrated_liquid_density',
                                  'long_name':'Vertically Integrated Liquid Density', 
                                  'coordinates':'elevation azimuth range'}, replace_existing=True)
    
    return radar

def calc_WV(radar = None, zh_name = 'corrected_reflectivity', h0c = None, wv_name = 'Waldvogel'):
    """
    Retrieves the height difference between the 0 °C isoterm and the maximum height of 45 dBZ for each gate, i.e., delta_H = H_45 - H_0 as in Waldvogel et al. (1979).
    In order for this function to work, all azimuths and elevations (if not constant, as with some brazilian radar)
    must be sorted following an increasing order.
    Number of azimuths between sweeps also must have the same shape. If not, it's recommended to use the sort_radar function
    which interpolates along rays to maintain the same azimutal shape.
    
    References
    __________
    
    Waldvogel, A., Federer, B., & Grimm, P. (1979). Criteria for the Detection of Hail Cells. Journal of Applied Meteorology and Climatology, 18(12), 1521–1525.
    Parameters
    
    Parameters
    __________
    
    radar: Radar 
        Radar object used
    zh_name: str
        Name of the horizontal reflectivity field factor. Default is corrected_reflectivity
    h0c: int of float
        height of the 0 °C isoterm in m.
    wg_name: str
        name used for the Echo Tops field. Default is Waldvogel.
        
    Returns
    __________
    
    Radar object with the ET field included (m)
    
    """

    # get all the sweeps' azimuths along and store them on a list for later. Also get the number of rays or azimuths of each sweep.
    azi = [radar.azimuth['data'][x] for x in radar.iter_slice()]
    n_rays = azi[0].shape[0]
    
    # get the elevation angles of each sweep. Sort them from the lowest to highest
    ele = radar.fixed_angle['data']
    sort_idx = np.argsort(ele)
    
    # get the number of sweeps
    n_sweeps = radar.nsweeps
    
    # get the radar range at the lowest elevation and the number of gates along each ray
    rg = radar.range['data']
    n_bins = radar.ngates

    # create empty matrixes to stack the reflectivity, x, y and z coordinates of each gate for every sweep. Also create an empty matrix
    # to store the heights between gates and the eventual echo tops
    ref_stacked = np.zeros((n_sweeps, n_rays, n_bins))
    x_stacked = np.zeros_like(ref_stacked)
    y_stacked = np.zeros_like(ref_stacked)
    z_stacked = np.zeros_like(ref_stacked) + radar.altitude['data']
    WV = np.zeros((n_rays, n_bins))
    
    # iterate over each sweep and stack the reflectivity, x, y, and z coordinates
    for i in sort_idx:
        ref_stacked[i, :, :] = radar.get_field(i, 'corrected_reflectivity', copy = True)
        x, y, z = radar.get_gate_x_y_z(i)
        x_stacked[i, :, :] = x
        y_stacked[i, :, :] = y
        z_stacked[i, :, :] = z
    
    # get the range of each gate
    r = np.sqrt(x_stacked**2 + y_stacked**2)
    
    # create a mask to remove gates below the desired threshold
    valid = ((z_stacked >= h0c) & (ref_stacked >= 45))

    # loop over every ray seeking for valid gates
    for az_idx in range(n_rays):
        slice_valid = valid[:, az_idx, :]
        if not np.any(slice_valid):
            continue
        # keep the valid gates
        slice_H45_elements = z_stacked[:, az_idx, :]
        # loop over every range gate
        for rg_idx in range(n_bins):
            H45_temp = 0
            sfc_rg = rg[rg_idx]
            #loop over every sweep 
            for el_idx in range(n_sweeps):
                if not slice_valid[el_idx, rg_idx]:
                    continue
                if slice_H45_elements[el_idx, rg_idx] == 0:
                    continue
                # if the first sweep, keep the gate heights
                if el_idx == 0:
                    H45_temp = slice_H45_elements[el_idx, rg_idx]
                # if not, keep iterating along every ray of every sweep
                else:
                # this is done in order to fit higher tilt rays with the nearest ray of the first sweep
                    ppi_ray_rg = r[el_idx, az_idx, :]
                    closest_idx = np.argmin(np.abs(ppi_ray_rg - sfc_rg))
                    # once the gate heights of each sweep are stored, look for the highest ones
                    H45_temp = np.nanmax(slice_H45_elements[el_idx, closest_idx])
            # once the iteration is finished, subtract the 45 dBZ heights by the 0 °C heights and store them in the lowest sweep
            if H45_temp > 0:
                WV[az_idx, rg_idx] = H45_temp - h0c
            
    # mask gates where the index is equal to zero
    WV = np.ma.masked_equal(WV, 0)
    
    # add 0 values to higher sweeps
    WV_field = np.zeros_like(radar.fields['corrected_reflectivity']['data'])
    WV_field[radar.get_slice(sort_idx[0])] = WV
    
    # add the Waldvogel field to the radar
    radar.add_field('Waldvogel', dic = {'units':'m',
                                        'data': WV_field, 'standard_name':'Waldvogel',
                                        'long_name':'Waldvogel', 
                                        'coordinates':'elevation azimuth range'}, replace_existing=True)

    return radar