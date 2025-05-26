#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.h"
#include "hittable.h"

#define MAX_OBJECTS 64 // Ajusta según tus necesidades

class hittable_list : public hittable {
public:
    hittable* objects[MAX_OBJECTS];
    int object_count;

    __host__ __device__
        hittable_list() : object_count(0) {}

    __host__ __device__
        hittable_list(hittable** objs, int n) : object_count(n) {
        for (int i = 0; i < n; ++i) objects[i] = objs[i];
    }
    __host__ __device__
        void clear() { object_count = 0; }

    __host__ __device__
        void add(hittable* object) {
        if (object_count < MAX_OBJECTS) {
            objects[object_count++] = object;
            bbox = aabb(bbox, object->bounding_box());
        }
    }

    __host__ __device__
        bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < object_count; ++i) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __host__ __device__
        aabb bounding_box() const override { return bbox; }

private:
    aabb bbox;
};

#endif
