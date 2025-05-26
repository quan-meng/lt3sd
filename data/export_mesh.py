import json
import numpy as np
import os
import h5py
import trimesh
import tyro
import glob
import dataclasses
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from third_parties.front3d_toolbox.scripts.utils import read_mesh_attr, transform_v
from tools.submit_utils import Slurm, submit_jobs
from tools.mesh_utils import merge_meshes
from tools.logger import get_logger
from configs.dataset import Front3D

logger = get_logger(file_name=__file__, debug="export_mesh")

FUTURE_PATH: str = "/cluster/daidalos/qmeng/Datasets/3D-FRONT/3D-FUTURE-model"
TEXTURE_PATH: str = "/cluster/daidalos/qmeng/Datasets/3D-FRONT/3D-FRONT-texture"
JSON_DIR: str = "/cluster/daidalos/qmeng/Datasets/3D-FRONT/3D-FRONT"
ROOT_DIR: str = Front3D.root_dir

selected_furniture_classes = [
    "Structure",
    "Cabinet/Shelf/Desk",
    "Bed",
    "Lighting",
    "Table",
    "Chair",
    "Sofa",
    "Pier/Stool",
]


class Furniture:
    def __init__(self, uid, jid, super_category, category, size=None, bbox=None):
        self.uid = uid
        self.jid = jid
        self.size = size
        self.bbox = bbox
        self.category = category
        self.super_category = super_category


class Instance:
    def __init__(self, f: Furniture, pos, rot, scale):
        self.info = f
        self.pos = pos
        self.rot = rot
        self.scale = scale


class Mesh:
    def __init__(self, uid, xyz, faces, uv, material, type, constructid, instanceid):
        self.uid = uid
        self.xyz = xyz
        self.faces = faces
        self.uv = uv
        self.material = material
        self.type = "Structure"

        self.constructid = constructid
        self.instanceid = instanceid


class Room:
    def __init__(self, type, instanceid):
        self.type = type
        self.instanceid = instanceid
        self.furniture = []
        self.meshes = {}
        self.center = 0

    def add_furniture(self, finstance):
        self.furniture.append(finstance)

    def add_mesh(self, mesh):
        if mesh.constructid not in self.meshes:
            self.meshes[mesh.constructid] = []
        self.meshes[mesh.constructid].append(mesh)

    def cal_center(self):
        for constructid, all_mesh in self.meshes.items():
            for mesh in all_mesh:
                if mesh.type == "floor":
                    v, faces, vt, mat = read_mesh_attr(mesh)

                    self.center = (np.max(v, 0) + np.min(v, 0)) / 2
                    return

    def output_meshes(self, select_mesh_type):
        all_output = []

        for _, all_mesh in self.meshes.items():
            for mesh in all_mesh:
                if len(select_mesh_type) > 0:
                    if mesh.type not in select_mesh_type:
                        continue

                idx = 0
                if mesh.type not in selected_furniture_classes:
                    logger.info(f"new mesh type: {mesh.type}")
                    # raise ValueError
                else:
                    idx = selected_furniture_classes.index(mesh.type)

                color = [idx, idx, idx, 1]

                v, faces, _, _ = read_mesh_attr(mesh)

                mesh = trimesh.Trimesh(vertices=v - self.center, faces=faces)
                mesh.visual.vertex_colors = np.tile(
                    np.array(color), (mesh.vertices.shape[0], 1)
                )
                all_output.append(mesh)

        return all_output

    def output_furniture(self, select_furniture_type, output_semantic_bbox=False):
        all_output, all_categories = [], []
        for instance in self.furniture:
            if instance.info.super_category not in selected_furniture_classes:
                # logger.info(f"new furniture type: {instance.info.super_category}")
                pass

            if len(select_furniture_type) > 0:
                if (
                    instance.info.category not in select_furniture_type
                    and instance.info.super_category not in select_furniture_type
                ):
                    continue

            obj_path = FUTURE_PATH + "/" + instance.info.jid + "/raw_model.obj"
            mesh = trimesh.load(obj_path, process=False)
            mesh.vertices = transform_v(mesh.vertices, instance)
            # mesh.visual.vertex_colors = np.tile(np.array(color), (mesh.vertices.shape[0], 1))

            all_output.append(mesh)
            all_categories.append(instance.info.super_category)

        if output_semantic_bbox:
            all_bboxes = []
            for mesh, category in zip(all_output, all_categories):
                bbox = np.array(mesh.bounding_box.bounds)  # [2, 3]
                all_bboxes.append((bbox, category))

            return all_output, all_bboxes
        else:
            return all_output, None

    def output(
        self,
        center,
        select_room_type,
        select_mesh_type,
        select_furniture_type,
        output_semantic_bbox=False,
    ):
        if center:
            self.cal_center()
        if len(select_room_type) > 0:
            if self.type not in select_room_type:
                return

        structure_meshes = self.output_meshes(select_mesh_type)
        furniture_meshes, all_bboxes = self.output_furniture(
            select_furniture_type, output_semantic_bbox=output_semantic_bbox
        )

        if len(structure_meshes) == 0 and len(furniture_meshes) == 0:
            logger.info("no mesh in room: ", self.instanceid)
            mesh = all_bboxes = None
        else:
            mesh = merge_meshes(structure_meshes + furniture_meshes)

        return mesh, all_bboxes


class House:
    def __init__(self, uid):
        self.uid = uid
        self.rooms = {}

    def add_room(self, room):
        self.rooms[room.instanceid] = room

    def del_room_by_id(self, room_id):
        return self.rooms.pop(room_id)

    def output_rooms(
        self,
        savepath,
        select_room_type,
        select_mesh_type,
        select_furniture_type,
        output_semantic_bbox=False,
    ):
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        for _, room in self.rooms.items():
            try:
                mesh, bboxes = room.output(
                    False,
                    select_room_type,
                    select_mesh_type,
                    select_furniture_type,
                    output_semantic_bbox=output_semantic_bbox,
                )

                # mesh.export(savepath + '/' + room.instanceid + '.obj')
                # np.save(savepath + '/' + room.instanceid + '_bbox.npy', bboxes)
                if bboxes is not None:
                    with h5py.File(
                        savepath + "/" + room.instanceid + "_bbox.h5", "w"
                    ) as h5file:
                        h5file.create_dataset(
                            "bbox", data=np.stack([x[0] for x in bboxes])
                        )
                        h5file.create_dataset(
                            "category", data=np.string_([x[1] for x in bboxes])
                        )
            except:
                logger.info(f"Error: {room.instanceid}")

    def output(
        self,
        savepath,
        select_room_type,
        select_mesh_type,
        select_furniture_type,
        output_semantic_bbox=False,
        add_floor=False,
    ):
        mesh, all_bboxes = [], []

        for _, room in self.rooms.items():
            try:
                mesh_i, bboxes_i = room.output(
                    False,
                    select_room_type,
                    select_mesh_type,
                    select_furniture_type,
                    output_semantic_bbox=output_semantic_bbox,
                )

                if mesh_i is not None:
                    mesh.append(mesh_i)
                    all_bboxes += bboxes_i
            except:
                logger.info(f"Error: {room.instanceid}")

        if len(mesh) > 0:
            mesh = merge_meshes(mesh)

            if add_floor:
                mesh = self.add_floor(mesh)

            mesh.export(savepath + ".obj")

            with h5py.File(savepath + "_bbox.h5", "w") as h5file:
                h5file.create_dataset("bbox", data=np.stack([x[0] for x in all_bboxes]))
                h5file.create_dataset(
                    "category", data=np.string_([x[1] for x in all_bboxes])
                )

    def add_floor(self, mesh):
        bounds_min = np.min(mesh.vertices, axis=0)
        bounds_max = np.max(mesh.vertices, axis=0)

        # add a rectangle floor [x_min, y_min, z_min] to [x_max, y_min, z_max] to the mesh
        floor = np.array(
            [
                [bounds_min[0], bounds_min[1], bounds_min[2]],
                [bounds_max[0], bounds_min[1], bounds_min[2]],
                [bounds_max[0], bounds_min[1], bounds_max[2]],
                [bounds_min[0], bounds_min[1], bounds_max[2]],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(
            vertices=np.vstack([mesh.vertices, floor]),
            faces=np.vstack([mesh.faces, faces + len(mesh.vertices)]),
        )

        return mesh


class Scene:
    def __init__(self, uid):
        self.furniture = {}
        self.meshes = {}
        self.material = {}
        self.house = House(uid)

    def add_furniture(self, f):
        self.furniture[f.uid] = f

    def add_mesh(self, m):
        self.meshes[m.uid] = m

    def output_rooms(
        self,
        savepath,
        select_room_type=[],
        select_mesh_type=[],
        select_furniture_type=[],
        output_semantic_bbox=True,
    ):
        os.makedirs(savepath, exist_ok=True)
        self.house.output_rooms(
            savepath + "/" + self.house.uid,
            select_room_type,
            select_mesh_type,
            select_furniture_type,
            output_semantic_bbox=output_semantic_bbox,
        )

    def output_houses(
        self,
        savepath,
        select_room_type=[],
        select_mesh_type=[],
        select_furniture_type=[],
        output_semantic_bbox=True,
        add_floor=False,
    ):
        os.makedirs(savepath, exist_ok=True)
        self.house.output(
            savepath + "/" + self.house.uid,
            select_room_type,
            select_mesh_type,
            select_furniture_type,
            output_semantic_bbox=output_semantic_bbox,
            add_floor=add_floor,
        )


def read_json(jsonfile) -> Scene:
    model_info = json.load(
        open(FUTURE_PATH + "/model_info.json", "r", encoding="utf-8")
    )
    model_info_dict = {}
    for model in model_info:
        model_info_dict[model["model_id"]] = model
    fid = open(jsonfile, "r", encoding="utf-8")
    try:
        data = json.load(fid)
    except:
        logger.info(f"Error in read_json: {jsonfile}")
        return None

    fid.close()
    scene = Scene(data["uid"])

    if "furniture" in data:
        for furniture in data["furniture"]:
            if "valid" in furniture and furniture["valid"]:
                f = Furniture(
                    furniture["uid"],
                    furniture["jid"],
                    model_info_dict[furniture["jid"]]["super-category"],
                    model_info_dict[furniture["jid"]]["category"],
                    furniture["size"] if "size" in furniture else None,
                    furniture["bbox"] if "bbox" in furniture else None,
                )
                scene.add_furniture(f)

    if "mesh" in data:
        for mesh in data["mesh"]:
            # Filter certain mesh types
            if "Slab" in mesh["type"]:
                continue

            m = Mesh(
                mesh["uid"],
                np.reshape(mesh["xyz"], [-1, 3]),
                np.reshape(mesh["faces"], [-1, 3]),
                np.reshape(mesh["uv"], [-1, 2]) if "uv" in mesh else None,
                (
                    scene.material[mesh["material"]]
                    if "material" in mesh and mesh["material"] in scene.material
                    else None
                ),
                mesh["type"],
                mesh["constructid"] if "constructid" in mesh else None,
                mesh["instanceid"] if "instanceid" in mesh else None,
            )
            scene.add_mesh(m)

    for r in data["scene"]["room"]:
        # Filter certain room types
        # if r['type'] in ['OtherRoom', 'Corridor', 'Balcony', 'Bathroom', 'Kitchen', 'LaundryRoom', 'Aisle', 'Hallway', 'StorageRoom']:
        #     continue

        room = Room(r["type"], r["instanceid"])

        children = r["children"]
        num_furniture = 0
        for c in children:
            ref = c["ref"]

            if ref in scene.furniture:
                finstance = Instance(
                    scene.furniture[ref], c["pos"], c["rot"], c["scale"]
                )
                room.add_furniture(finstance)

                num_furniture += 1

            if ref in scene.meshes:
                room.add_mesh(scene.meshes[ref])

        # Filter rooms with no furniture
        if num_furniture > 0:
            scene.house.add_room(room)

    return scene


def export_rooms(slurm: Slurm, output_semantic_bbox: bool = False):
    save_dir = os.path.join(ROOT_DIR, "3D-FRONT-room")
    fn_kwargs_list = [
        {"path": x} for x in sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    ]

    def func(path: str):
        try:
            scene = read_json(path)
            (
                scene.output_rooms(save_dir, output_semantic_bbox=output_semantic_bbox)
                if scene is not None
                else None
            )
        except Exception as e:
            logger.info(f"An error occurred: {e}, path: {path}")

    submit_jobs(
        fn=func,
        slurm_kwargs=dataclasses.asdict(slurm),
        fn_kwargs_list=fn_kwargs_list,
    )


def export_houses(
    slurm: Slurm, output_semantic_bbox: bool = True, add_floor: bool = True
):
    save_dir = os.path.join(ROOT_DIR, "3D-FRONT-house")
    fn_kwargs_list = [
        {"path": x} for x in sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    ]

    def func(path: str):
        try:
            scene = read_json(path)
            (
                scene.output_houses(
                    save_dir,
                    output_semantic_bbox=output_semantic_bbox,
                    add_floor=add_floor
                )
                if scene is not None
                else None
            )
        except Exception as e:
            logger.info(f"An error occurred: {e}, path: {path}")

    submit_jobs(
        fn=func,
        slurm_kwargs=dataclasses.asdict(slurm),
        fn_kwargs_list=fn_kwargs_list,
    )


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "export_rooms": export_rooms,
            "export_houses": export_houses,
        }
    )
