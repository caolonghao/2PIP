from basicpy import BaSiC
import jax
import numpy as np

class BaSicEstimate:
    def __init__(self, debug=False, extra_dark_field = False) -> None:
        jax.config.update("jax_platform_name", "cpu")
        self.basic = BaSiC(get_darkfield=True, smoothness_flatfield=1)
        self.debug = debug
        self.extra_dark_field = extra_dark_field
        
    def __call__(self, img_stack):
        if self.extra_dark_field:
            # self.basic.fit(img_stack, darkfield=None)
            bg_value = np.min(img_stack)
            bg = np.ones_like(img_stack[0]) * bg_value
            self.basic.fit(img_stack - bg)
            
            return self.basic.flatfield, bg

        else:
            self.basic.fit(img_stack)
            return self.basic.flatfield, self.basic.darkfield