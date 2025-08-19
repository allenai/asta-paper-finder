class SortedSet(set):
    def __iter__(self):
        return iter(sorted(super().__iter__()))
