import warnings


class SuitableInput:
    def __init__(self, datm, section):
        self.datm = datm
        self.section = section
        b, r = self.__calc()
        if not b:
            warnings.warn("\nYour chose section was [" + self.section + "] and this is not suitable to you time data.\n"
                          "Your time data have an interval of [" + str(r) + " " + self.section + "]")

    def __calc(self):
        if self.datm.__len__() > 1:
            diff = (self.datm[1].to_pydatetime() - self.datm[0].to_pydatetime()).total_seconds()
            if self.section == "year":
                return diff / (3600 * 24 * 365) <= 1.0, diff / (3600 * 24 * 365)
            elif self.section == "month":
                return diff / (3600 * 24 * 30) <= 1.0, diff / (3600 * 24 * 30)
            elif self.section == "day":
                return diff / (3600 * 24) <= 1.0, diff / (3600 * 24)
            elif self.section == "hour":
                return diff <= 3600, diff / 3600
            else:
                raise Exception("The section [" + self.section + "] is not defined in "+str(self))