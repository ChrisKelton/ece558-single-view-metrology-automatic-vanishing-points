__all__ = [
    "ImageTexture",
    "Appearance",
    "TextureCoordinate",
    "Coordinate",
    "IndexedFaceSet",
    "Shape",
    "Collision",
    "Object3D",
    "write_vrml_file",
    "write_3d_vrml_file_with_default_options",
]
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, List, Any

import numpy as np


@dataclass
class ImageTexture:
    url: Union[Path, str]

    def asdict(self) -> Dict[str, Union[Path, str]]:
        return {"url": self.url}


@dataclass
class Appearance:
    texture: ImageTexture
    material: Dict[str, List[float]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.material = {"diffuseColor": [0.7, 0.7, 0.7]}

    def asdict(self) -> Dict[str, Any]:
        return {
            "texture ImageTexture": self.texture.asdict(),
            "material DEF mat Material": self.material,
        }


@dataclass
class TextureCoordinate:
    point: np.ndarray

    def asdict(self) -> Dict[str, np.ndarray]:
        return {"point": self.point}


@dataclass
class Coordinate:
    point: np.ndarray

    def asdict(self) -> Dict[str, np.ndarray]:
        return {"point": self.point}


@dataclass
class IndexedFaceSet:
    coord: Coordinate
    coordIndex: List[int]
    texCoord: TextureCoordinate
    texCoordIndex: List[int]
    solid: bool = False

    def asdict(self) -> Dict[str, Any]:
        return {
            "coord Coordinate": self.coord.asdict(),
            "coordIndex": self.coordIndex,
            "texCoord TextureCoordinate": self.texCoord.asdict(),
            "texCoordIndex": self.texCoordIndex,
            "solid": self.solid.__str__().upper(),
        }


@dataclass
class Shape:
    appearance: Appearance
    geometry: IndexedFaceSet

    def asdict(self) -> Dict[str, Any]:
        return {
            "appearance Appearance": self.appearance.asdict(),
            "geometry IndexedFaceSet": self.geometry.asdict(),
        }


@dataclass
class Collision:
    children: List[Shape]
    collide: bool = False

    def asdict(self) -> Dict[str, Any]:
        children_list = []
        for child in self.children:
            children_list.append({"Shape": child.asdict()})

        return {"collide": self.collide.__str__().upper(), "children": children_list}


@dataclass
class Object3D:
    collision: Collision

    def asdict(self) -> Dict[str, Any]:
        return {"Collision": self.collision.asdict()}


def write_dict_to_txt_file(dict_: Union[dict, list], f, indent=1):
    str_indent = ""
    for indent_cnt in range(np.clip(len(dict_) - indent, 0, 50)):
        str_indent += "\t"
    if isinstance(dict_, dict):
        for key in dict_.keys():
            f.write(f"{str_indent}{key} ")
            val = dict_.get(key)
            if isinstance(val, dict):
                f.write("{\n")
                write_dict_to_txt_file(val, f, indent + 1)
                f.write("%s}" % str_indent)
            elif isinstance(val, bool) or isinstance(val, str):
                if val == "FALSE" or val == "TRUE":
                    f.write(f"{val}\n")
                else:
                    f.write(f'"{val}"\n')
            elif isinstance(val, list) or isinstance(val, np.ndarray):
                f.write("[\n")
                write_dict_to_txt_file(val, f, indent + 1)
                f.write(f"{str_indent}]\n")
    elif isinstance(dict_, np.ndarray):
        # for cnt, val in enumerate(dict_):
        #     if isinstance(val, dict):
        #         write_dict_to_txt_file(val, f, indent)
        #     elif isinstance(val, np.ndarray):
        val = dict_.copy()
        arr_shape = val.shape
        if val.ndim == 2:
            for row in range(arr_shape[0]):
                for col in range(arr_shape[1]):
                    if col == arr_shape[1] - 1:
                        f.write(f"{val[row, col]}")
                    elif col == 0:
                        f.write(f"{str_indent}{val[row, col]} ")
                    else:
                        f.write(f"{val[row, col]} ")
                f.write(",\n")
        elif val.ndim == 1:
            for idx in range(arr_shape[0]):
                if idx == arr_shape[0] - 1:
                    f.write(f"{val[idx]}")
                elif idx == 0:
                    f.write(f"{str_indent}{val[idx]} ")
                else:
                    f.write(f"{val[idx]} ")
            f.write(",\n")
    elif isinstance(dict_, list):
        for cnt, val in enumerate(dict_):
            if isinstance(val, dict):
                write_dict_to_txt_file(val, f, indent)
            elif cnt == 0:
                f.write(f"{str_indent}{val} ")
            elif cnt == len(dict_) - 1:
                f.write(f"{val}")
            else:
                f.write(f"{val} ")
        f.write(",\n")


def write_vrml_file(obj: Union[dict, Object3D], output_file: Path):
    if isinstance(obj, Object3D):
        obj = obj.asdict()

    with open(output_file, "w+") as f:
        f.write("#VRML V2.0 utf8\n\n")
        write_dict_to_txt_file(obj, f)


def write_3d_vrml_file_with_default_options(
    xy_texture: str, yz_texture: str, zx_texture: str, output_wrl_file: Union[Path, str]
):
    appearanceXY: Appearance = Appearance(ImageTexture(xy_texture))
    xy_coord = [
        [0.0, 0.0, 10.0],
        [14.0, 0.0, 10.0],
        [14.0, 26.0, 10.0],
        [0.0, 26.0, 10.0],
    ]
    xy_texcoord = [[-0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.0, 1.0]]
    geometryXY = IndexedFaceSet(
        coord=Coordinate(np.array(xy_coord)),
        coordIndex=[0, 1, 2, 3, -1],
        texCoord=TextureCoordinate(np.array(xy_texcoord)),
        texCoordIndex=[0, 1, 2, 3, -1],
    )
    xy_shape = Shape(appearance=appearanceXY, geometry=geometryXY)

    appearanceZX = Appearance(ImageTexture(zx_texture))
    zx_coord = [[0.0, 0.0, 0.0], [14.0, 0.0, 0.0], [14.0, 0.0, 10.0], [0.0, 0.0, 10.0]]
    zx_texcoord = [[-0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.0, 1.0]]
    geometryZX = IndexedFaceSet(
        coord=Coordinate(np.array(zx_coord)),
        coordIndex=[0, 1, 2, 3, -1],
        texCoord=TextureCoordinate(np.array(zx_texcoord)),
        texCoordIndex=[0, 1, 2, 3, -1],
    )
    zx_shape = Shape(appearance=appearanceZX, geometry=geometryZX)

    appearanceYZ = Appearance(ImageTexture(yz_texture))
    yz_coord = [[0.0, 26.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 26.0, 10.0]]
    yz_texcoord = [[-0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.0, 1.0]]
    geometryYZ = IndexedFaceSet(
        coord=Coordinate(np.array(yz_coord)),
        coordIndex=[0, 1, 2, 3, -1],
        texCoord=TextureCoordinate(np.array(yz_texcoord)),
        texCoordIndex=[0, 1, 2, 3, -1],
    )
    yz_shape = Shape(appearance=appearanceYZ, geometry=geometryYZ)

    vrml_collision = Collision(children=[xy_shape, zx_shape, yz_shape])
    object_3d = Object3D(vrml_collision).asdict()

    write_vrml_file(object_3d, output_wrl_file)


def main():
    appearanceXY: Appearance = Appearance(ImageTexture("XY-cropped.JPG"))
    xy_coord = [
        [0.0, 0.0, 10.0],
        [14.0, 0.0, 10.0],
        [14.0, 26.0, 10.0],
        [0.0, 26.0, 10.0],
    ]
    xy_texcoord = [[-0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.0, 1.0]]
    geometryXY = IndexedFaceSet(
        coord=Coordinate(np.array(xy_coord)),
        coordIndex=[0, 1, 2, 3, -1],
        texCoord=TextureCoordinate(np.array(xy_texcoord)),
        texCoordIndex=[0, 1, 2, 3, -1],
    )
    xy_shape = Shape(appearance=appearanceXY, geometry=geometryXY)

    appearanceZX = Appearance(ImageTexture("ZX-cropped.JPG"))
    zx_coord = [[0.0, 0.0, 0.0], [14.0, 0.0, 0.0], [14.0, 0.0, 10.0], [0.0, 0.0, 10.0]]
    zx_texcoord = [[-0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.0, 1.0]]
    geometryZX = IndexedFaceSet(
        coord=Coordinate(np.array(zx_coord)),
        coordIndex=[0, 1, 2, 3, -1],
        texCoord=TextureCoordinate(np.array(zx_texcoord)),
        texCoordIndex=[0, 1, 2, 3, -1],
    )
    zx_shape = Shape(appearance=appearanceZX, geometry=geometryZX)

    appearanceYZ = Appearance(ImageTexture("YZ-cropped.JPG"))
    yz_coord = [[0.0, 26.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 26.0, 10.0]]
    yz_texcoord = [[-0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.0, 1.0]]
    geometryYZ = IndexedFaceSet(
        coord=Coordinate(np.array(yz_coord)),
        coordIndex=[0, 1, 2, 3, -1],
        texCoord=TextureCoordinate(np.array(yz_texcoord)),
        texCoordIndex=[0, 1, 2, 3, -1],
    )
    yz_shape = Shape(appearance=appearanceYZ, geometry=geometryYZ)

    vrml_collision = Collision(children=[xy_shape, zx_shape, yz_shape])
    object_3d = Object3D(vrml_collision).asdict()

    output_wrl_file = Path("./test-object.wrl")
    write_vrml_file(object_3d, output_wrl_file)


if __name__ == "__main__":
    main()
