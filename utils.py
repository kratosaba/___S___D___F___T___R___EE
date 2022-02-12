def save_experiment(name_txt_file,parameters):
    
    with open(name_txt_file, 'w') as f: 
        for key, value in parameters.items(): 
            f.write('%s:%s\n' % (key, value))

