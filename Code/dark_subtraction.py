def subtract_dark(data, masterdark, masterflat_normed):
    """Subtracts darkness from raw corrected data, assuming master dark and flat (the latter normed)
    frames are known"""
    return (data - masterdark)/masterflat_normed
