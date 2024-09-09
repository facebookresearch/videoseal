import omegaconf

omegaconf.OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
omegaconf.OmegaConf.register_new_resolver("add", lambda x, y: x + y)

# in the yaml, allows for
# vae:
#   msg_processor:
#     nbits: 16
#     hidden_size: ${mul:${vae.msg_processor.nbits},2}
