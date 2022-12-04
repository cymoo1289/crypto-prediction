import yaml
creds = "config.yaml"
with open(creds) as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)
 
config['credentials'] = {'usernames': {'moo': {'name': 'MOO', 'password': '$2b$12$GndD9vJFvOjIFkYXW79YaOPvqaVPHRNSo2bZ3nGja8NqaBGxL.pFC'}, 'moo2': {'name': 'MOO2', 'password': '$2b$12$hgcBpZWNxm/9X87Fz89tuOhzT6hzYDAoOAcDX6.EZ8dA.iD5x2OxW'}, 'xx': {'name': 'xx', 'password': '$2b$12$7xG5GghTbrMmRJful5foHuhhKJ2taQmp769FFau5gyWSYKEBU.p46', 'email': 'xx'}}}
print(config)
with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)