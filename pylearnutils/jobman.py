from jobman.tools import DD

def extract_everything(train_obj):
    dd = DD(channels.items())
    channels = train_obj.model.monitor.channels
    for k in channels.keys():
        values = channels[k].val_record
        dd[k] = values
    return dd
