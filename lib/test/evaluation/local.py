from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/got10k_lmdb'
    settings.got10k_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/itb'
    settings.lasot_extension_subset_path_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/lasot_lmdb'
    settings.lasot_path = '/run/user/1000/gvfs/smb-share:server=192.168.1.1,share=vot/LaSOT/'
    settings.network_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/nfs'
    settings.otb_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/otb'
    settings.prj_dir = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker'
    settings.result_plot_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/output/test/result_plots'
    settings.results_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/output'
    settings.segmentation_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/output/test/segmentation_results'
    settings.tc128_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/trackingnet'
    settings.uav_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/uav'
    settings.vot18_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/vot2018'
    settings.vot22_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/vot2022'
    settings.vot_path = '/home/mohamadalansari/Desktop/CLDTrack/Train/CLDTracker/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

