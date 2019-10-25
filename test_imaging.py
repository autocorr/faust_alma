#!/usr/bin/env python
"""
============
Image Target
============
Create continuum and spectral line images for the FAUST target CB68.

NOTE: Run with `execfile` in CASA to use this script.
"""
from __future__ import (print_function, division)

import os
import shutil
import datetime
from glob import glob
from collections import (namedtuple, OrderedDict)

import numpy as np


ROOT_DIR = '/lustre/naasc/users/bsvoboda/faust_cb68/'
DAT_DIR = ROOT_DIR + 'data/'
IMG_DIR = DAT_DIR + 'images/'
VIS_FILES = {
        ('S1', 'TM1'): [DAT_DIR+'CB68_a_06_TM1/uid___A002_Xd5c881_Xf288.ms'],
        ('S1', 'TM2'): [DAT_DIR+'CB68_a_06_TM2/uid___A002_Xd88143_X4922.ms'],
        ('S1', '7M'):  [DAT_DIR+'CB68_a_06_7M/uid___A002_Xd3e89f_X8ba5.ms'],
        ('S2', 'TM1'): [DAT_DIR+'CB68_b_06_TM1/uid___A002_Xd66bb8_X6418.ms'],
        ('S2', 'TM2'): [DAT_DIR+'CB68_b_06_TM2/uid___A002_Xd88143_X4bea.ms', DAT_DIR+'CB68_b_06_TM2/uid___A002_Xd88143_X9508.ms'],
        ('S2', '7M'):  [DAT_DIR+'CB68_b_06_7M/uid___A002_Xd2d9a0_X8d8b.ms'],
        ('S3', 'TM1'): None,
        ('S3', 'TM2'): [DAT_DIR+'CB68_a_03_TM2/uid___A002_Xd64dca_X66b6.ms'],
}


Spw = namedtuple('Spw',
        'name, restfreq, spw_id, ot_name,'
        'nchan, chan_width, tot_bw, bbc_num, line_win, fitchan')

SETUP1_SPWS = OrderedDict( [(spw_.name, spw_) for spw_ in [
    Spw(         'DCO_3_2', '216112.582MHz',  0, 'ALMA_RB_06#BB_1#SW-01#FULL_RES',  480, 122.056,   58586.9, 1, 64, '10~29;153~188;307~470'),
    Spw(    'NH2D_322_312', '216562.716MHz',  1, 'ALMA_RB_06#BB_1#SW-02#FULL_RES',  480, 122.056,   58586.9, 1, 64, '10~60;128~169;267~413;435~456'),
    Spw(         'SiO_5_4', '217104.919MHz',  2, 'ALMA_RB_06#BB_1#SW-03#FULL_RES',  480, 122.056,   58586.9, 1, 64, '10~163;265~470'),
    Spw(  'c-C3H2_606_515', '217822.148MHz',  3, 'ALMA_RB_06#BB_1#SW-04#FULL_RES',  480, 122.056,   58586.9, 1, 64, '10~207;272~470'),
    Spw(    'H2CO_303_202', '218222.192MHz',  4, 'ALMA_RB_06#BB_2#SW-01#FULL_RES',  480, 122.056,   58586.9, 2, 64, '10~36;46~188;294~403;432~470'),
    Spw(   'CH3OH_422_312', '218440.063MHz',  5, 'ALMA_RB_06#BB_2#SW-02#FULL_RES',  480, 122.056,   58586.9, 2, 64, '10~103;141~201;298~348;392~439'),
    Spw(        'C18O_2_1', '219560.354MHz',  6, 'ALMA_RB_06#BB_2#SW-03#FULL_RES',  480, 122.056,   58586.9, 2, 64, '10~125;182~199;284~436'),
    Spw(        'SO_65_54', '219949.442MHz',  7, 'ALMA_RB_06#BB_2#SW-04#FULL_RES',  480, 122.056,   58586.9, 2, 64, '10~102;149~191;281~338;371~470'),
    Spw(       'OCS_19_18', '231060.993MHz',  8, 'ALMA_RB_06#BB_3#SW-01#FULL_RES',  480, 122.056,   58586.9, 3, 64, '10~27;52~175;294~350;374~470'),
    Spw(        '13CS_5_4', '231220.685MHz',  9, 'ALMA_RB_06#BB_3#SW-02#FULL_RES',  480, 122.056,   58586.9, 3, 64, '24~37;66~195;311~470'),
    Spw(        'DN2p_3_2', '231321.8283MHz',10, 'ALMA_RB_06#BB_3#SW-03#FULL_RES',  480, 122.056,   58586.9, 3, 64, '10~80;100~136;169~288;344~470'),
    Spw(    'D2CO_404_303', '231410.234MHz', 11, 'ALMA_RB_06#BB_3#SW-04#FULL_RES',  480, 122.056,   58586.9, 3, 64, '10~65;79~175;205~208;274~285;370~470'),
    Spw(           'cont0', '233795.75MHz',  12, 'ALMA_RB_06#BB_4#SW-01#FULL_RES', 3840, 488.224, 1874781.7, 4, -1, '27~126;207~598;828~1069;1172~1266;1390~1457;1726~1777;2149~2345;2407~2586;2666~3333;3425~3539;3632~3716'),
]])

SETUP2_SPWS = OrderedDict( [(spw_.name, spw_) for spw_ in [
    Spw(   'CH3OH_514_413', '243915.788MHz',  0, 'ALMA_RB_06#BB_1#SW-01#FULL_RES',  512, 122.076,   62502.7,  1,  64, '10~216;283~355;412~506'),
    Spw(    'H2CS_716_615', '244048.504MHz',  1, 'ALMA_RB_06#BB_1#SW-02#FULL_RES',  512, 122.076,   62502.7,  1,  64, '10~210;335~500'),
    Spw(          'CS_5_4', '244935.557MHz',  2, 'ALMA_RB_06#BB_1#SW-03#FULL_RES',  512, 122.076,   62502.7,  1,  64, '10~231;288~500'),
    Spw(      'HC3N_27_26', '245606.320MHz',  3, 'ALMA_RB_06#BB_1#SW-04#FULL_RES',  512, 122.076,   62502.7,  1,  64, '37~210;335~500'),
    Spw(           'cont1', '246700.000MHz',  4, 'ALMA_RB_06#BB_2#SW-01#FULL_RES', 2048, 976.605, 2000087.7,  2,  -1, '97~190;252~447;522~778;827~1030;1079~1428;1455~1967'),
    Spw(    'NH2CHO_12_11', '260189.848MHz',  5, 'ALMA_RB_06#BB_3#SW-01#FULL_RES',  512, 122.076,   62502.7,  3,  64, '10~133;159~505'),
    Spw(   'HCOOCH3_21_20', '260255.080MHz',  6, 'ALMA_RB_06#BB_3#SW-02#FULL_RES',  512, 122.076,   62502.7,  3,  64, '10~209;248~502'),
    Spw(  'CH2DOH_524_515', '261692.000MHz',  7, 'ALMA_RB_06#BB_3#SW-03#FULL_RES',  512, 122.076,   62502.7,  3,  64, '10~326;335~500'),
    Spw(         'CCH_3_2', '262004.260MHz',  8, 'ALMA_RB_06#BB_3#SW-04#FULL_RES',  512, 122.076,   62502.7,  3,  64, '24~208;272~505'),
    Spw(  'CH2DOH_423_313', '257895.6727MHz', 9, 'ALMA_RB_06#BB_4#SW-01#FULL_RES',  512, 122.076,   62502.7,  4,  64, '10~210;335~500'),
    Spw(        'SO_66_55', '258255.8259MHz',10, 'ALMA_RB_06#BB_4#SW-02#FULL_RES',  512, 122.076,   62502.7,  4,  64, '20~228;286~502'),
    Spw(   'CH3OCH3_14_13', '258548.819MHz', 11, 'ALMA_RB_06#BB_4#SW-03#FULL_RES',  512, 122.076,   62502.7,  4,  64, '10~147;182~314;329~500'),
    Spw(    'HDCO_422_321', '259034.910MHz', 12, 'ALMA_RB_06#BB_4#SW-04#FULL_RES',  512, 122.076,   62502.7,  4,  64, '28~46;101~197;226~244;280~365;404~494'),
]])

SETUP3_SPWS = OrderedDict( [(spw_.name, spw_) for spw_ in [
    Spw(        'N2Hp_1_0',  '93180.9MHz',    0, 'ALMA_RB_03#BB_1#SW-01#FULL_RES',  960,  61.032,   58591.0,  1,  64, '50~500;640~950'),
    Spw('13CH3OH_2-12_1-11', '94407.129MHz',  1, 'ALMA_RB_03#BB_1#SW-02#FULL_RES',  960,  61.032,   58591.0,  1,  64, '20~221;316~950'),
    Spw(           'cont2',  '94999.908MHz',  2, 'ALMA_RB_03#BB_2#SW-01#FULL_RES', 3840, 488.258, 1874911.2,  2,  -1, '10~3736;3755~3830'),
    Spw(   'CH3OH_313_404', '107013.831MHz',  3, 'ALMA_RB_03#BB_3#SW-01#FULL_RES',  960,  61.032,   58591.0,  1,  64, '20~635;660~940'),
    Spw(       'C3D_11_9f', '108064.0MHz',    4, 'ALMA_RB_03#BB_3#SW-02#FULL_RES',  960,  61.032,   58591.0,  1,  64, '11~474;514~832;853~910'),
    Spw(  'SO2_1019_10010', '104239.2952MHz', 5, 'ALMA_RB_03#BB_4#SW-01#FULL_RES',  960,  61.032,   58591.0,  1,  64, '10~23;55~950'),
    Spw(    'H13C3N_12_11', '105799.113MHz',  6, 'ALMA_RB_03#BB_4#SW-02#FULL_RES',  960,  61.032,   58591.0,  1,  64, '10~950'),
]])

SPWS = {'S1': SETUP1_SPWS, 'S2': SETUP2_SPWS, 'S3': SETUP3_SPWS}


Target = namedtuple('Target', 'name, ra, dec, field, vlsr')

TARGETS = { targ.name : targ for targ in [
    Target('CB68',
        #'16:57:19.643', '-16.09.23.920', '0', 5.0),  # Proposal coordinates
        '16:57:19.647', '-16.09.23.950', '0', 5.0),  # Continuum Gaussian fit from Muneaki
]}


ArrayConfig = namedtuple('ArrayConfig', 'imsize, cell, gridder')

# Array configurations are calculated for HPBW/4.5 cell and 2.44*L/D (FWBN) size
ARRAY_CONFIGS = {
        ('S1', 'TM1'): ArrayConfig([ 615,  615], '0.094arcsec', 'standard'),  # 0.422 &  57.7
        ('S1', 'TM2'): ArrayConfig([ 164,  164], '0.352arcsec', 'standard'),  # 1.583 &  57.7
        ('S1', '7M'):  ArrayConfig([  78,   78], '1.267arcsec', 'standard'),  # 5.70  &  98.9
        ('S1', 'JNT'): ArrayConfig([ 630,  630], '0.094arcsec', 'mosaic'),    # (dup of TM1)
        ('S2', 'TM1'): ArrayConfig([ 615,  615], '0.083arcsec', 'standard'),  # 0.374 &  51.1
        ('S2', 'TM2'): ArrayConfig([ 164,  164], '0.311arcsec', 'standard'),  # 1.40  &  51.1
        ('S2', '7M'):  ArrayConfig([  78,   78], '1.122arcsec', 'standard'),  # 5.05  &  87.6
        ('S2', 'JNT'): ArrayConfig([ 630,  630], '0.083arcsec', 'mosaic'),    # (dup of TM1)
        ('S3', 'TM1'): ArrayConfig([1836, 1836], '0.064arcsec', 'standard'),  # 0.290 & 117.5
        ('S3', 'TM2'): ArrayConfig([ 403,  403], '0.291arcsec', 'standard'),  # 1.31  & 117.5
        ('S3', 'JNT'): ArrayConfig([1836, 1836], '0.064arcsec', 'mosaic'),    # (dup of TM1)
}


class ImagingConfig(object):
    cs_ext = '.contsub'

    def __init__(self, targ, setup=None, array=None, contsub=False):
        """
        Parameters
        ----------
        targ : Target
        setup : str
        array : str
        contsub : bool
        """
        assert setup in ('S1', 'S2', 'S3')
        assert array in ('TM1', 'TM2', '7M', 'JNT')
        self.targ = targ
        self.setup = setup
        self.array = array
        self.contsub = contsub
        self.array_config = ARRAY_CONFIGS[(setup, array)]
        if array == 'JNT':
            self.vis = []
            for s in ('TM1', 'TM2', '7M'):
                try:
                    vis = VIS_FILES[(setup, s)]
                    self.vis.extend(vis)
                except KeyError:
                    pass
        else:
            self.vis = VIS_FILES[(setup, array)]
        if contsub:
            self.vis = [s + self.cs_ext for s in self.vis]

    def get_basename(self, spw, ext=None):
        base = 'images/{0}/{0}_{1}_{2}_{3}'.format(self.targ.name, self.setup, self.array, spw.name)
        if ext is None:
            return base
        else:
            return '{0}_{1}'.format(base, ext)

    def get_spw_id(self, spw):
        """
        The SPW ID's may not be consistent between the configurations or
        different execution blocks. This method uses the MSMD tool to get the
        name assigned by the OT to the baseline-board pair (BlBP) for use as a
        unique identifier and converts it to numerical SPW ID number.
        """
        spw_ids = []
        for vis in self.vis:
            msmd.open(vis)
            labels = msmd.namesforspws()
            obs_spws = msmd.spwsforintent('OBSERVE_TARGET#ON_SOURCE')
            id_str = ','.join([
                    str(i) for i,s in enumerate(labels)
                    if i in obs_spws and s.endswith(spw.ot_name)
            ])
            spw_ids.append(id_str)
            msmd.close()
        if not spw_ids:
            raise ValueError('Empty ID Name list: {0}'.format(spw_ids))
        else:
            return spw_ids


###############################################################################
# General utility functions
###############################################################################

def log_post(msg):
    """
    Post a message to the CASA logger and logfile.
    """
    casalog.post(msg, 'INFO', 'bsvoboda')


def check_delete_image_files(imagename, parallel=False, preserve_mask=False):
    """
    Check for and remove (if they exist) files created by clean such as '.flux',
    '.image', etc.
    NOTE this function has issues with deleting tables made by clean in
    parallel mode, probably because the directory structure is different.

    Parameters
    ----------
    imagename : str
        The relative path name to the files to delete.
    parallel : bool, default False
        rmtables can't remove casa-images made with parallel=True, they must be
        manually removed.
    preserve_mask : bool, default False
        Whether to preserve the `.mask` file extension
    """
    log_post(':: Check for and remove existing files')
    exts = [
        '.flux', '.pb', '.image', '.weight', '.model', '.pbcor', '.psf',
        '.sumwt', '.residual', '.flux.pbcoverage',
    ]
    if not preserve_mask:
        exts += ['.mask']
    # CASA image table directories
    for ext in exts:
        filen = imagename + ext
        if os.path.exists(filen):
            if parallel:
                log_post('-- Hard delete {0}'.format(ext))
                shutil.rmtree(filen)
            else:
                log_post('-- Removing {0}'.format(filen))
                rmtables(filen)
    # "Cannot delete X because it's not a table" -> so hard delete
    for ext in ('.residual', '.workdirectory'):
        filen = imagename + ext
        if os.path.exists(filen):
            log_post('-- Hard delete {0}'.format(ext))
            shutil.rmtree(filen)


def delete_all_extensions(imagename, keep_exts=None):
    """
    Parameters
    ----------
    imagename : str
    keep_exts : None, iterable
        A list of extensions to keep, example: ['mask', 'psf']
    """
    for filen in glob(imagename+'.*'):
        if keep_exts is not None and any(filen.endswith(ext) for ext in keep_exts):
            continue
        try:
            log_post(':: Removing {0}'.format(filen))
            rmtables(filen)
            shutil.rmtree(filen)
            log_post('-- Hard Delete!')
        except OSError:
            pass


def export_fits(imagename, overwrite=True):
    log_post(':: Exporting fits')
    exportfits(imagename, imagename+'.fits', velocity=True, overwrite=overwrite)


def if_exists_remove(imagename):
    if os.path.exists(imagename):
        rmtables(imagename)


def export_fits_all():
    for targ in TARGETS.keys():
        for mol in SPWS.keys():
            log_post(':: Export {0}_{1} to FITS'.format(targ, mol))
            imagename = IMG_DIR + '{0}/{0}_{1}.image'.format(targ, mol)
            export_fits(imagename)


def clean_workdirs_all():
    for targ in TARGETS.keys():
        for path in glob(IMG_DIR+'{0}/{0}_*.workdirectory'.format(targ)):
            log_post(':: Hard delete {0}'.format(path))
            shutil.rmtree(path)


def primary_beam_correct(imagebase, overwrite=True, export=True):
    log_post(':: Check for and remove existing files')
    imagename  = imagebase + '.image'
    pbimage    = imagebase + '.pb'
    pbcorimage = imagebase + '.pbcor'
    impbcor(imagename=imagename, pbimage=pbimage, outfile=pbcorimage,
            overwrite=overwrite)
    if export:
        export_fits(pbcorimage)


def calc_start_velo(targ, spw):
    window = float(spw.velo_width.strip('km/s')) * spw.nchan
    start_velo = '{0:.4f}km/s'.format(targ.vlsr - window / 2)
    return start_velo


def concat_parallel_image(imagename):
    outfile = imagename + '.concat'
    if_exists_remove(outfile)
    ia.open(imagename)
    im_tool = ia.imageconcat(
            outfile=outfile,
            infiles=imagename+'/*.image',
            reorder=True,
    )
    im_tool.close()
    ia.done()


def convert_to_common_beam(imagename):
    outfile = imagename + '.combm'
    if_exists_remove(outfile)
    imsmooth(imagename, kernel='common', outfile=outfile)


def get_spw_list(targ, kind='line'):
    spws = SPWS[targ.name]
    if kind == 'line':
        return [s for s in spws.values() if not s.name.startswith('cont')]
    elif kind == 'cont':
        return [s for s in spws.values() if s.name.startswith('cont')]
    elif kind == 'all':
        return [s for s in spws.values()]
    else:
        raise ValueError('Unknown kind: "{0}"'.format(kind))


def fitspw_from_spws(spws):
    spw_list = spws.values() if isinstance(spws, dict) else spws
    return ','.join('{0}:{1}'.format(s.spw_id, s.fitchan) for s in spw_list)


###############################################################################
# Calibration
###############################################################################

def do_statwt_all():
    """
    The pipeline task `hifv_statwt` was removed from the script, so the
    measurement sets need to have their weight spectrums computed excluding the
    channels with line emission.

    NOTE: this must be run using CASA 5.4.1 (pre-release as of 10/23/18) in
    order to have the `fitspw` keyword argument.
    """
    log_post(':: Apply statwt to all measurement sets')
    for vis in VIS_FILES:
        log_post('-- {0}'.format(os.path.basename(vis)))
        statwt(vis=vis, fitspw=BASELINE_CHANS)


def split_test_data():
    """
    Split out a small test data set on one EB, one target, and a few channels
    around the brightest NH3 (1,1) line.
    """
    outputvis = ROOT_DIR + 'test_imaging/test_split_1eb.ms'
    targ = TARGETS['NGC1333IRAS4A']
    spw = '{0}:236~276'.format(SPWS[targ.name]['NH3_11'].spw_id)
    split(
        vis=get_vis_name(targ),
        outputvis=outputvis,
        field=targ.name,
        spw=spw,
    )


def get_vis_name(targ, contsub=False, pipe='default'):
    assert pipe in ('default', 'claire')
    cs_ext = '.contsub' if contsub else ''
    vis = DAT_DIR + 'pipe_{0}/'.format(pipe) + targ.ms_name + cs_ext
    return vis


###############################################################################
# Line imaging
###############################################################################

def test_clean_line_target(cfg, spw, ext=None, iterzero=False, fullcube=False,
        parallel=False):
    """
    Run `tclean` on a target. Uses multi-scale clean and auto-masking.

    Parameters
    ----------
    cfg : ImagingConfig
    spw : Spw
    ext : string, default None
        Extension to add to image basename
        Example: 'try2' for "G28539_nh3_11_try2" base
    iterzero : bool, default False
    fullcube : bool, default False
        Image the entire cube or a window/channel-subset around the line
    parallel : bool, default True
        Whether to use tclean in parallel mode (i.e. with mpicasa). Note that
        there is a bug on the non-latest versions of CASA where tclean in
        parallel mode will abort because of an incorrect file path created for
        the work-directory.
    """
    imagename = cfg.get_basename(spw, ext=ext)
    spw_id = cfg.get_spw_id(spw)
    log_post(':: Running clean for {0}'.format(imagename))
    niter = 0 if iterzero else int(1e6)
    if not fullcube and spw.line_win != -1:
        start = int(spw.nchan / 2) - spw.line_win
        nchan = spw.line_win * 2
    else:
        start = -1
        nchan = -1
    delete_all_extensions(imagename)
    tclean(
        vis=cfg.vis,
        imagename=imagename,
        field=cfg.targ.name,
        spw=spw_id,
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start,
        nchan=nchan,
        imsize=cfg.array_config.imsize,
        cell=cfg.array_config.cell,
        # gridder parameters
        gridder=cfg.array_config.gridder,
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 9, 18, 36],  # delta, 1, 2, 4 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=1.0,
        niter=niter,
        nsigma=2.0,
        pblimit=-0.001,
        interactive=False,
        parallel=parallel,
        # automasking parameters
        usemask='auto-multithresh',  # ALMA joint values
        noisethreshold=4.25,
        sidelobethreshold=2.0,
        lownoisethreshold=1.0,
        minbeamfrac=0.3,
        pbmask=0.0,
        negativethreshold=0.0,
        verbose=True,
        pbcor=True,
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)


def clean_all_lines(cfg):
    for ii, spw in enumerate(SPWS[cfg.setup]):
        test_clean_line_target(cfg, spw, parallel=True)


def pbcor_all_lines(targ, overwrite=True):
    log_post(':: Primary beam correcting maps')
    for spw in get_spw_list(targ, kind='line'):
        log_post('-- Correcting {0}_{1}'.format(targ.name, spw.name))
        imagebase = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
        imagename  = imagebase + '.image'
        pbimage    = imagebase + '.pb'
        pbcorimage = imagebase + '.pbcor'
        impbcor(imagename=imagename, pbimage=pbimage, outfile=pbcorimage,
                overwrite=overwrite)


###############################################################################
# Continuum imaging and subtraction
###############################################################################

def clean_line_with_uvtaper(cfg, spw, fullcube=False, parallel=True):
    """
    Run tclean on a target with a uv-taper but without multiscale clean. This
    resulting image is used to create a mask for later procedures.

    Parameters
    ----------
    targ : ImagingConfig
    spw : Spw
    fullcube : bool, default False
        Image the full cube or a subset of channels about the system velocity.
    parallel : bool, default True
        Whether to use tclean in parallel mode (i.e. with mpicasa). Note that
        there is a bug on the non-latest versions of CASA where tclean in
        parallel mode will abort because of an incorrect file path created for
        the work-directory.
    """
    log_post(':: Running clean with uv-taper ({0}, {1})'.format(targ.name, spw.name))
    imagename = cfg.get_basename(spw, ext='uvtaper')
    spw_id = cfg.get_spw_id(spw)
    # restart parameters
    if not fullcube and spw.line_win != -1:
        start = spw.nchan // 2 - spw.line_win
        nchan = spw.line_win * 2
    else:
        start = -1
        nchan = -1
    delete_all_extensions(imagename)
    tclean(
        vis=cfg.vis,
        imagename=imagename,
        field=targ.name,
        spw=spw_id,
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start,
        nchan=nchan,
        imsize=cfg.array_config.imsize,
        cell=cfg.array_config.cell,
        # gridder parameters
        gridder='standard',
        uvtaper='5arcsec',
        # deconvolver parameters
        deconvolver='hogbom',
        restoringbeam='common',
        weighting='briggs',
        robust=2.0,
        niter=int(1e6),
        nsigma=2.0,
        interactive=False,
        parallel=parallel,
        # automasking parameters
        usemask='auto-multithresh',  # use ALMA 12m(short) values
        noisethreshold=3.0,
        sidelobethreshold=3.0,
        lownoisethreshold=2.0,
        minbeamfrac=0.3,
        negativethreshold=1000.0,
        verbose=True,
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)


def convert_to_commonbeam(imagename):
    log_post(':: Smoothing spatial axes')
    outfile = imagename + '.common'
    if_exists_remove(outfile)
    imsmooth(
            imagename=imagename,
            outfile=outfile,
            kernel='commonbeam',
    )


def spectral_smooth(imagename, n_hanning=9):
    """
    Smooth the cube by repeatedly applying Hanning smooth operations.
    Smooth the cube by a set number of channels without decimation.

    NOTE the number of channels should be odd to ensure that there are
    channels that match the input dataset exactly in frequency

    Parameters
    ----------
    imagename : str
    n_hanning : number
        Number of times to apply Hanning spectral smooth operation
    """
    log_post(':: Smoothing spectral axis')
    infile = imagename + '.common'
    outfile = imagename + '.specsmooth'
    tmpfile = outfile + '.tmp'
    if_exists_remove(outfile)
    if_exists_remove(tmpfile)
    shutil.copytree(infile, tmpfile)
    for ii in range(n_hanning):
        log_post('-- Hanning smooth iteration: {0}'.format(ii))
        specsmooth(
                imagename=tmpfile,
                outfile=outfile,
                function='hanning',
                dmethod='',  # no decimation
        )
        rmtables(tmpfile)
        os.rename(outfile, tmpfile)
    os.rename(tmpfile, outfile)


def mask_from_smooth(imagename, thresh=5.0e-3):
    log_post(':: Creating mask data from smoothed cube')
    mask_ext = 'smooth'
    # mask manipulation syntax cannot have path syntax, must just be filename
    # alone so change directory when calculating the mask.
    workdir = os.path.dirname(imagename)
    basedir = os.path.abspath('.')
    basename = os.path.basename(imagename)
    os.chdir(workdir)
    smooth_image = basename + '.specsmooth'
    smooth_image_masked = smooth_image + '.masked'
    if_exists_remove(smooth_image_masked)
    shutil.copytree(smooth_image, smooth_image_masked)
    ia.open(smooth_image_masked)
    ia.calcmask(
            mask='{0} > {1}'.format(smooth_image_masked, thresh),
            name=mask_ext,
    )
    ia.done()
    os.chdir(basedir)
    log_post(':: Creating mask file')
    outfile = imagename + '.smask'
    if_exists_remove(outfile)
    smooth_image = imagename + '.specsmooth.masked'
    makemask(
            mode='copy',
            inpimage=smooth_image,
            inpmask='{0}:{1}'.format(smooth_image, mask_ext),
            output=outfile,
    )


def mask_from_uvtaper(cfg, spw):
    """
    Create mask from uv-tapered image cube: convert to common beam, spectrally
    smooth, and generate mask from smoothed cube.
    """
    imagebase = cfg.get_basename(spw, ext='uvtaper')
    imagename = imagebase + '.image'
    convert_to_commonbeam(imagename)
    spectral_smooth(imagename)
    mask_smoothed_image(imagename)


def clean_line_with_uvt_mask(cfg, spw, fullcube=False, parallel=True):
    """
    Run tclean on a target with a mask genearted froom a uv-tapered and
    spectrally smoothed initial imaging. A simple hogbom clean is used without
    auto-multithresh or multi-scale.

    Parameters
    ----------
    targ : ImagingConfig
    spw : Spw
    fullcube : bool, default False
        Image the full cube or a subset of channels about the system velocity.
    parallel : bool, default True
        Whether to use tclean in parallel mode (i.e. with mpicasa). Note that
        there is a bug on the non-latest versions of CASA where tclean in
        parallel mode will abort because of an incorrect file path created for
        the work-directory.
    """
    log_post(':: Running clean with uv-taper ({0}, {1})'.format(targ.name, spw.name))
    imagename = cfg.get_basename(spw, ext='smask')
    maskname = cfg.get_basename(spw, ext='uvtaper') + '.image.smask'
    spw_id = cfg.get_spw_id(spw)
    # restart parameters
    if not fullcube and spw.line_win != -1:
        start = spw.nchan // 2 - spw.line_win
        nchan = spw.line_win * 2
    else:
        start = -1
        nchan = -1
    delete_all_extensions(imagename)
    tclean(
        vis=cfg.vis,
        imagename=imagename,
        field=targ.name,
        spw=spw_id,
        specmode='cube',
        outframe='lsrk',
        veltype='radio',
        restfreq=spw.restfreq,
        start=start,
        nchan=nchan,
        imsize=cfg.array_config.imsize,
        cell=cfg.array_config.cell,
        # gridder parameters
        gridder='standard',
        # deconvolver parameters
        deconvolver='multiscale',
        scales=[0, 5, 10],  # point, 1, 2 beam hpbw's
        smallscalebias=0.6,
        restoringbeam='common',
        weighting='briggs',
        robust=2.0,
        niter=int(1e6),
        nsigma=2.0,
        interactive=False,
        parallel=parallel,
        # mask from smoothed uv-taper
        usemask='user',
        mask=maskname,
        verbose=True,
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)




###############################################################################
# Continuum imaging and subtraction
###############################################################################

def uvcontsub_target(targ):
    vis = get_vis_name(targ)
    spws = SPWS[targ.name]
    fitspw = fitspw_from_spws(spws)
    spw = ','.join(str(s.spw_id) for s in spws.values())
    uvcontsub(
        vis=vis,
        spw=spw,
        fitspw=fitspw,
        fitorder=0,
        solint='int',
    )


def test_clean_cont_target(targ, ext=None, iterzero=False, restart=False,
        parallel=False):
    """
    Run tclean on a target to generate images of the continuum. Uses
    multi-scale, multi-term, aw-project, and automasking.

    Parameters
    ----------
    targ : Target
    spw : Spw
    ext : string, default None
        Extension to add to imagename
        Example: 'joint' for 'G28539_nh3_11_joint' base
    iterzero : bool, default False
        If True, set niter=0 and create a dirty map
    restart : bool, default False
        If restarting, do not calculate the PSF or residual files
    parallel : bool, default False
    """
    log_post(':: Running clean ({0}, continuum)'.format(targ.name))
    imagename = 'images/{0}/{0}_contfull'.format(targ.name)
    if ext is not None:
        imagename = '{0}_{1}'.format(imagename, ext)
    # restart parameters
    niter = 0 if iterzero else int(1e6)
    calcpsf = not restart  # ie True when not restarting
    calcres = not restart
    cont_spws = get_spw_list(targ, kind='cont')
    linefree = fitspw_from_spws(cont_spws)
    #spws = '40:5~60'  # XXX
    delete_all_extensions(imagename)
    tclean(
        vis=[get_vis_name(targ)],
        imagename=imagename,
        field=targ.name,
        spw=linefree,
        specmode='mfs',
        nterms=2,
        imsize=[1472, 1472],  # 189.3 as, efficient size?
        cell='0.1286arcsec',  # 0.9 as / 7
        # gridder parameters
        gridder='standard',
        # deconvolver parameters
        deconvolver='mtmfs',
        scales=[0, 7, 14],  # point, 1, 2 beam hpbw's
        smallscalebias=0.6,
        weighting='briggs',
        robust=1.0,
        niter=niter,
        threshold='9.0uJy',  # 30uJy for one SPW, 5 uJy for all SPWs XXX
        interactive=True,
        parallel=parallel,
        # automasking parameters
        usemask='auto-multithresh',
        noisethreshold=5.0,
        sidelobethreshold=3.0,
        lownoisethreshold=2.0,
        minbeamfrac=0.3,
        negativethreshold=0.0,
        verbose=True,
        # restart parameters
        restart=restart,
        calcpsf=calcpsf,
        calcres=calcres,
        #savemodel='modelcolumn',
    )
    workdir = '{0}.workdirectory'.format(imagename)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)


def delmod_target(targ):
    vis = get_vis_name(targ)
    if isinstance(vis, list):
        for filen in vis:
            delmod(vis=filen)
    else:
        delmod(vis=vis)


def gaincal_cont_target(targ, calt_ext='pcal1'):
    log_post(':: Running gaincal ({0}, continuum)'.format(targ.name))
    vis = get_vis_name(targ)
    caltable = '{0}.{1}'.format(vis, calt_ext)
    cont_spws = get_spw_list(targ, kind='cont')
    linefree = fitspw_from_spws(cont_spws)
    gaincal(
        vis=vis,
        field=targ.name,
        caltable=caltable,
        spw=linefree,
        solint='inf',
        combine='spw',
        refant='ea02',
        minblperant=4,
        minsnr=3,
        gaintype='G',
        calmode='p',
        append=False,
    )


###############################################################################
# Moment maps
###############################################################################

def make_ia_moments(targ, spw):
    outfile_fmt = 'moments/{name}/{name}_{line}_snr{snr}_smooth{smooth}'
    maxv = 200  # sigma, need upper bound for range argument
    smoothaxes = [0, 1, 3]  # ra, dec, velo
    smoothtypes = ['gauss', 'gauss', 'hann']
    # FWHM of kernel in 0.1286 as pixels. 14 pix -> 1.800 as
    # 3 pixels for hanning smooth of spectral
    smoothwidths = [14, 14, 3]
    velowidth = 3  # km/s, radius of window
    xypix = (1500, 1500)
    region = (
            'box[[0pix,0pix],[{0}pix,{1}pix]], '.format(*xypix) +
            'range=[{0:.2f}km/s,{1:.2f}km/s]'.format(
                    targ.vlsr - velowidth, targ.vlsr + velowidth)
    )
    imagename_base = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
    imagename = imagename_base + '.image'
    # scale typical RMS for NH3 (1,1) to other windows based on channel width
    rms = float(calc_threshold(spw, nsigma=1).rstrip('mJy')) / 1e3
    ia.open(imagename)
    for snr in (-10, 1, 2, 3, 4):
        # unsmoothed moments
        ia.moments(
            moments=[-1,0,1,2,8],
            region=region,
            axis=3,
            includepix=[snr*rms,maxv],
            outfile=outfile_fmt.format(
                    name=targ.name, line=spw.name, snr=str(snr),
                    smooth='0'),
            overwrite=True,
        ).done()
        # smoothed moments
        # the RMS will change from smoothing, so modify threshold
        # ratio of beam sizes in pixels times a sqrt(2) factor from
        # Hanning smooth in velocity
        eta = smoothwidths[0] / 7.0 * np.sqrt(2)
        ia.moments(
            moments=[-1,0,1,2,8],
            region=region,
            axis=3,
            includepix=[snr*rms/eta,maxv],
            smoothaxes=smoothaxes,
            smoothtypes=smoothtypes,
            smoothwidths=smoothwidths,
            outfile=outfile_fmt.format(
                name=targ.name, line=spw.name, snr=str(snr),
                smooth=str(smoothwidths[0])),
            overwrite=True,
        ).done()
    ia.close()


def make_all_moments():
    for targ in TARGETS.values():
        log_post(':: Calculating all moments for {0}'.format(targ.name))
        for spw in get_spw_list(targ, kind='line'):
            log_post('-- Moments {0}_{1}'.format(targ.name, spw.name))
            make_ia_moments(targ, spw)


def pbcor_moments(targ, overwrite=True):
    log_post(':: Primary beam correcting moment maps')
    for spw in get_spw_list(targ, kind='line'):
        log_post('-- Correcting {0}_{1}'.format(targ.name, spw.name))
        pbimage = 'images/{0}/{0}_{1}.pb'.format(targ.name, spw.name)
        # NOTE We need to remove spectral axis in the primary beam image in
        # order for both data to have the same shape.  Luckily, `impbcor` can
        # be fed an array as well as a file name.
        ia.open(pbimage)
        pbdata = ia.getregion()[...,0]  # beam of channel number 0
        ia.close()
        infiles = glob('moments/{0}/{0}_{1}_*'.format(targ.name, spw.name))
        infiles = [
                s for s in infiles
                if s.endswith('average')
                or s.endswith('integrated')
                or s.endswith('maximum')
        ]
        for imagename in infiles:
            impbcor(imagename=imagename, pbimage=pbdata,
                    outfile=imagename+'.pbcor', overwrite=overwrite)


def pbcor_all_moments():
    for targ in TARGETS.values():
        pbcor_moments(targ)


def export_moments(targ):
    infiles = [
        s for s in glob('moments/{0}/{0}_*'.format(targ.name))
        if not s.endswith('.fits')
    ]
    for imagename in infiles:
        export_fits(imagename)


def export_all_moments():
    for targ in TARGETS.values():
        export_moments(targ)


###############################################################################
# One off tests
###############################################################################

def test_imspec_smooth(targ, spw):
    imagebase = 'images/{0}/{0}_{1}'.format(targ.name, spw.name)
    imagename = imagebase + '.image'
    imagemask = imagebase + '_smmask.image'
    # get data from original image
    ia.open(imagebase+'.image')
    data = ia.getchunk()
    ia.close()
    # copy image
    shutil.rmtree(imagemask)
    shutil.copytree(imagename, imagemask)
    # create mask for point source
    rms_fullres  = imstat(imagename, chans='10')
    mask_fullres = data > rms_fullres
    # spatial smooth
    # spectral smooth
    # compute rms
    # create mask
    # put data into new image
    ia.open(imagebase+'_smmask.image')
    ia.putchunk(data)
    ia.close()


