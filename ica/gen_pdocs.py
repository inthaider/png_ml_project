import pdoc

options = {
    'output_directory': 'docs_modules',
    'force': True,
    'overwrite': True,
    'skip': ['__init__.py']
}

pdoc.render('modules', **options)
