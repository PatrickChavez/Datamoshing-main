from app import normalize_ai_params, parse_fraction
print(parse_fraction('30000/1001'), parse_fraction('0/0'), parse_fraction('24'))
print(normalize_ai_params({'effect':'delta_repeat','start_frame':10,'end_frame':5,'fps':900,'delta':100}, {'total_frames':60,'fps':30}))
