def getParameter(cfg, name, default_value):
  value = None
  if name in cfg.keys():
    value = cfg[name]
  else:
    value = default_value
  return value
