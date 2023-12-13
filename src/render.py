import seaborn as sns
import ipyvolume as ipv
import trimesh
import trimesh.transformations as trtf
import ipywidgets as widgets
from ipywidgets import HBox, VBox, interact, fixed
from IPython.display import clear_output
import numpy as np
import seaborn as sns
from .item import Item
from .bin3d import Bin, get_force_graph
from typing import Any
import networkx as nx
import matplotlib.pyplot as plt
import ray
import copy
from . import util


def _join_item_boxes(items):
    """Joins the boxes of the given items to one large mesh."""
    boxes = [item_to_mesh(i) for i in items]
    vertices, faces = [], []
    offset = 0
    for box in boxes:
        vertices.append(np.array(box.vertices))
        faces.append(np.array(box.faces)+offset)
        offset += len(box.vertices)
    return np.concatenate(vertices), np.concatenate(faces)


def _transform_colors_to_join_boxes(colors):
    """Transforms a list of colors to colors for a joint box mesh."""
    assert len(colors.shape) == 2
    shape_1 = colors.shape[1]
    n_vertices_per_box = 8
    colors = np.tile(colors, n_vertices_per_box).reshape(-1, shape_1)
    return colors


def render_env_3d(env: Any, **kwargs):
    """Renders a environment in 3D using a tab for each bin."""
    tab = widgets.Tab()
    tab.children = [render_bin_3d(b) for b in env.bins]
    for i, b in enumerate(env.bins):
        percent = int(np.round(b.volume_percentage()*100))
        title = f'#{i+1} - {b.name} ({percent}%)'
        tab.set_title(i, title)
    display(tab)


def render_env_2d(env, **kwargs):
    out = widgets.Output()
    a = widgets.Output()
    b = widgets.Output()
    with a:
        print(f'state:\n{env.state}')
        sns.heatmap(env.bin3d.heightmap, square=True, vmin=0, vmax=env.bin3d.extents[2], **kwargs)
        plt.show()
    with b:
        item = env.selection_item.get()
        print(f'selected item:\n{item}')
        if item is not None:
            sns.heatmap(np.full(item.extents[:2], item.extents[2]), vmin=0, vmax=env.bin3d.extents[2], square=True)
            plt.show()
    with out:
        display(widgets.HBox([a, b]))
    return out


def plot_place_probs(action_probs, env):
    plot = widgets.Output()
    slider = widgets.FloatSlider(
        value=1,
        min=0,
        max=action_probs['place'].max(),
        step=0.005,
        description='Max',
        continuous_update=False,
        orientation='vertical',
    )

    def heatmap(*args):
        with plot:
            clear_output(wait=True)
            plt.figure(figsize=(3, 3))
            plt.title('Place Probabilities')
            sns.heatmap(env.remove_padding(action_probs['place'].reshape(env.observation_space['heightmap'].shape[:2])),
                        vmin=0, vmax=slider.value, square=True)
            plt.tight_layout()
            plt.show()

    heatmap()
    slider.observe(heatmap, 'value')
    return widgets.HBox([plot, slider])


def plot_rotate_probs(action_probs):
    out = widgets.Output()
    with out:
        plt.figure(figsize=(3, 3))
        plt.title('Rotate Probabilities')
        sns.barplot(x=np.arange(len(action_probs['rotate'])), y=action_probs['rotate'])
        plt.tight_layout()
        plt.show()
    return out


def plot_probabilities(env, info, space):
    """Plots the given probabilites contained in info."""
    action_probs = util.get_action_probabilities(info, space)
    return widgets.HBox([plot_rotate_probs(action_probs), plot_place_probs(action_probs, env)])


def plot_policy(agent: Any, env: Any) -> (dict, widgets.Widget):
    """Plots the given agent's policy and environment."""
    policy = agent.get_policy()
    pp = ray.rllib.models.preprocessors.DictFlatteningPreprocessor(env.observation_space)
    space = copy.deepcopy(env.action_space)
    space.original_space = space
    obs = env.get_observation()
    action, state_out, info = policy.compute_single_action(pp.transform(obs))
    return action, widgets.VBox([
        render_env_2d(env),
        plot_probabilities(env, info, space)
    ])


def item_to_mesh(item: Item) -> trimesh.Trimesh:
    """Creates a mesh for the given Item."""
    position = item.position
    if position is None:
        position = [0, 0, 0]
    box = trimesh.creation.box(item.extents, trtf.translation_matrix(position))
    box = box.apply_translation(np.array(item.extents)/2)
    return box


def render_items_3d(items: list) -> list:
    """Renders the given items in 3D and returns the ipyvolume meshes."""
    if len(items) == 0:
        return widgets.Widget()
    vertices, faces = _join_item_boxes(items)
    mesh = ipv.plot_trisurf(*vertices.T, triangles=faces)
    mesh.material.transparent = True
    mesh.material.side = "FrontSide"
    return mesh


def render_force_graph_3d(bin3d: Bin) -> tuple:
    """Renders the force graph as lines and markers."""
    items = bin3d.items
    force_graph = get_force_graph(items)
    floor_xyz = np.concatenate([(bin3d.extents/2)[:2], [0]])
    xyz = np.stack([i.mean() for i in items] + [floor_xyz])
    idx = {item.id: i for i, item in enumerate(items)}
    idx['floor'] = len(xyz)-1
    lines = []
    for edge in force_graph.edges:
        lines.append([idx[edge[0]], idx[edge[1]]])
    # create plots
    if len(lines) > 0:
        force_tree_lines = ipv.plot_trisurf(*xyz.T, lines=lines, color='black')
    else:
        force_tree_lines = widgets.Widget()
    force_tree_markers = ipv.scatter(*xyz.T, size=30, marker='point_2d')
    return force_tree_lines, force_tree_markers


def render_bin_bounds_3d(bin3d: Bin):
    """Renders the bounds of the given bin in 3D."""
    ex, ey, ez = bin3d.extents
    lines = [
        # bottom and top rectangles and pillar in [0,0]
        np.array([[0, 0, 0], [ex, 0, 0], [ex, ey, 0], [0, ey, 0], [0, 0, 0], [0, 0, ez],
                  [ex, 0, ez], [ex, ey, ez], [0, ey, ez], [0, 0, ez]]).T,
        # pillars in [ex,0], [ex,ey] and [0,ey]
        np.array([[ex, 0, 0], [ex, 0, ez]]).T,
        np.array([[ex, ey, 0], [ex, ey, ez]]).T,
        np.array([[0, ey, 0], [0, ey, ez]]).T,
    ]
    for line in lines:
        ipv.plot(*line, color='brown')


def render_bin_3d(bin3d: Bin, palette: str = None) -> widgets.Output:
    """Renders the given bin and returns the rendering output."""
    out = widgets.Output()
    bin_3d_plot = Bin3DPlot(bin3d)
    force_graph_plot = ForceGraphPlot(bin3d)
    bin_3d_plot.step_slider.observe(force_graph_plot.on_slider_change, names='value')
    with out:
        display(
            VBox([
                HBox([bin_3d_plot.render()], layout={
                    'overflow': 'hidden',  # there is a very small scroll region, simply hide it
                    }),
                HBox([force_graph_plot.render()], layout={'border': '1px solid lightgray'}),
            ])
        )
    return out


class Bin3DPlot():
    """Class to render a Bin in 3D."""
    def __init__(self, bin3d: Bin):
        """Creates a new rendering instance for the given Bin."""
        self.bin3d = bin3d
        self.out = widgets.Output()
        # slider
        self.step_slider = widgets.IntSlider(min=0, max=len(bin3d.items), value=len(bin3d.items),
                                             layout={'width': '90%'}, description='Step', continuous_update=False)
        self.step_slider.observe(self.on_step_slider_change, names='value')
        self.alpha_slider = widgets.FloatSlider(min=0.0, max=1.0, value=1.0, layout={'width': '90%'},
                                                description='Alpha', continuous_update=False)
        self.alpha_slider.observe(self.on_alpha_slider_change, names='value')
        # buttons
        self.button_minus = widgets.Button(description='-')
        self.button_plus = widgets.Button(description='+')
        def on_button_minus_clicked(button): self.step_slider.value = self.step_slider.value - 1
        def on_button_plus_clicked(button): self.step_slider.value = self.step_slider.value + 1
        self.button_minus.on_click(on_button_minus_clicked)
        self.button_plus.on_click(on_button_plus_clicked)
        # dropdown
        self.color_dropdown = widgets.Dropdown(options=['none', 'force', 'pressure'], description='Color',
                                               layout={'width': '90%'})
        self.color_dropdown.observe(self.on_dropdown_change, names='value')
        self.color = 'none'
        # checkbox
        self.force_tree_checkbox = widgets.Checkbox(description='Force Tree')
        self.force_tree_checkbox.observe(self.on_force_tree_checkbox_change, names='value')

    def on_force_tree_checkbox_change(self, change):
        """On force_tree_checkbox change."""
        self.force_tree_lines.visible = self.force_tree_markers.visible = change['new']

    def on_step_slider_change(self, change):
        """On step_slider change."""
        self.set_color()

    def on_alpha_slider_change(self, change):
        """On alpha_slider change."""
        self.set_color()

    def on_dropdown_change(self, change):
        """On dropdown change."""
        self.color = change['new']
        self.set_color()

    def set_color(self):
        """Sets the meshes' color based on self.color setting."""
        if len(self.bin3d.items) == 0:
            return
        if self.color in ['force', 'pressure']:
            items = self.bin3d.items[:self.step_slider.value]
            force_graph = get_force_graph(items)
            colors = metric_to_colors(force_graph, [i.id for i in items], self.color)
        else:
            colors = sns.color_palette(None, len(self.bin3d.items))
        alpha = self.alpha_slider.value
        colors = np.concatenate([colors, np.full((len(colors), 1), alpha)], 1)
        # make non-selected boxes invisible
        colors[self.step_slider.value:, 3] = 0
        n_missing_items = len(self.bin3d.items) - len(colors)
        colors = np.concatenate([colors, np.zeros((n_missing_items, 4))])
        # transform to mesh
        colors = _transform_colors_to_join_boxes(colors)
        self.mesh.color = colors

    def render(self) -> widgets.Output:
        """Renders this rendering instance."""
        items = self.bin3d.items
        self.fig = ipv.figure(width=800, height=600)
        self.mesh = render_items_3d(items)
        # render force graph
        self.force_tree_lines, self.force_tree_markers = render_force_graph_3d(self.bin3d)
        self.force_tree_lines.visible = self.force_tree_markers.visible = self.force_tree_checkbox.value
        self.set_color()
        render_bin_bounds_3d(self.bin3d)
        palette_mesh = self.render_palette()
        ipv.xyzlim(0, max(self.bin3d.extents))
        ipv.style.box_off()
        ipv.style.axes_off()
        # layout
        with self.out:
            display(VBox([
                HBox([self.fig], layout={'overflow': 'hidden',  # there is a very small scroll region, simply hide it
                                         'height': '600px'}),   # prevent flickering
                HBox([
                    VBox([self.step_slider, self.alpha_slider, self.force_tree_checkbox], layout={'width': '75%'}),
                    VBox([HBox([self.button_minus, self.button_plus]), self.color_dropdown]),
                ])
            ]))
        return self.out

    def render_palette(self):
        """Renders a palette below the bin boundaries. The palette has the same extents in the xy-plane as the bin."""
        mesh = palette(*self.bin3d.extents[:2]*10)
        mesh = ipv.plot_trisurf(*np.array(mesh.vertices).T, triangles=np.array(mesh.faces),
                                color=np.array([173, 140, 103])/255)
        mesh.material.transparent = True
        mesh.material.side = "FrontSide"
        return mesh


class ForceGraphPlot():
    """Class to render a Bin as force graph."""
    def __init__(self, bin3d: Bin):
        """Creates a new rendering instance for the given Bin."""
        self.bin3d = bin3d
        self.item_plot_idx = len(bin3d.items)
        self.out = widgets.Output()
        self.node_color_dropdown = widgets.Dropdown(options=['item', 'force', 'pressure'], description='Node Color:')
        self.edge_label_dropdown = widgets.Dropdown(options=['none', 'force', 'pressure'], description='Edge Label:')
        self.node_color_dropdown.observe(self.on_dropdown_change, names='value')
        self.edge_label_dropdown.observe(self.on_dropdown_change, names='value')

    def on_dropdown_change(self, change):
        """On dropdown change."""
        self.render()

    def on_slider_change(self, change):
        """On slider change."""
        self.item_plot_idx = change['new']
        self.render()

    def render(self) -> widgets.Output:
        """Renders this instance."""
        force_graph = get_force_graph(self.bin3d.items[:self.item_plot_idx])
        plot_out = widgets.Output()
        with plot_out:
            # layout = nx.drawing.layout.multipartite_layout(force_graph, align='horizontal')
            layout = nx.drawing.nx_agraph.graphviz_layout(force_graph, prog='dot')
            layout = {k: -np.array(v) for k, v in layout.items()}  # flip layout
            node_labels = nx.get_node_attributes(force_graph, self.node_color_dropdown.value)
            node_labels = {k: np.round(v, 2) for k, v in node_labels.items()}
            edge_labels = nx.get_edge_attributes(force_graph, self.edge_label_dropdown.value)
            edge_labels = {k: np.round(v, 2) for k, v in edge_labels.items()}

            if self.node_color_dropdown.value == 'item':
                floor_color = np.array([0.8, 0.8, 0.8])
                # check if more than just floor node is in graph
                n_items = len(force_graph.nodes) - 1
                if n_items > 0:
                    palette = sns.color_palette(None, n_colors=n_items)
                    idxs = [force_graph.nodes[node]['idx'] for node in force_graph.nodes()]
                    colors = np.take(palette, idxs, axis=0)
                    colors[idxs.index(-1)] = floor_color  # set floor's (-1) color to floor_color
                else:
                    colors = floor_color.reshape(1, 3)
            else:
                colors = metric_to_colors(force_graph, force_graph.nodes, self.node_color_dropdown.value, norm_idx=-2)
            plt.figure(figsize=(14, 7))
            nx.draw(force_graph, labels=node_labels, node_color=colors, width=0.4, node_size=500, pos=layout)
            nx.draw_networkx_edge_labels(force_graph, edge_labels=edge_labels, pos=layout)
            plt.show()

        self.out.clear_output(wait=True)
        with self.out:
            display(VBox([
                HBox([self.node_color_dropdown, self.edge_label_dropdown]),
                plot_out
            ]))
        return self.out


def metric_to_colors(force_graph: nx.DiGraph, nodes: list, metric: str = None, n_colors: int = 100,
                     norm_idx: int = -1) -> list:
    """Returns a list of colors based on the given force graph, nodes and metric. Norm index is used
    to norm by another element than the maximum (which is typically the floor)."""
    if len(nodes) == 0:
        return np.zeros((0, 3))
    color_palette = sns.color_palette('coolwarm', n_colors)
    metrics = [force_graph.nodes[n][metric] for n in nodes]
    norm_value = np.sort(metrics)[norm_idx]
    if np.isclose(norm_value, 0):  # prevent divide by zero
        norm_value = 1.0
    metrics_normed = np.array(metrics) / norm_value
    metrics_color_idx = np.interp(metrics_normed, [0, 1], [0, n_colors-1]).round().astype(int)
    colors = np.take(color_palette, metrics_color_idx, axis=0)
    return colors


def palette(width_scaling: int = 800, length_scaling: int = 1200):
    """Creates a palette mesh. For measurements see: https://de.wikipedia.org/wiki/Europoolpalette"""
    width = 800
    length = 1200
    floor_outer = np.array([100, length, 22])
    floor_h = floor_outer[2]
    floor_mid = np.array([145, length, floor_h])
    offset_mid_x = (width/2-floor_mid[0]/2)

    def box(extents, position):
        return trimesh.creation.box(extents=extents).apply_translation(extents/2).apply_translation(position)
    # floors
    floor1 = box(floor_outer, [0, 0, 0])
    floor2 = box(floor_mid, [offset_mid_x, 0, 0])
    floor3 = box(floor_outer, [width-floor_outer[0], 0, 0])

    # pillars
    pillar = np.array([floor_outer[0], 145, 78])
    # outer outer
    pillar_1 = box(pillar, [0, 0, floor_h])
    pillar_2 = box(pillar, [width-floor_outer[0], 0, floor_h])
    pillar_3 = box(pillar, [0, length-pillar[1], floor_h])
    pillar_4 = box(pillar, [width-floor_outer[0], length-pillar[1], floor_h])
    # outer mid
    pillar_5 = box(pillar, [0, length/2-pillar[1]/2, floor_h])
    pillar_6 = box(pillar, [width-floor_outer[0], length/2-pillar[1]/2, floor_h])
    # mid outer and inner
    pillar_mid = np.array([145, 145, pillar[2]])
    pillar_7 = box(pillar_mid, [offset_mid_x, 0, floor_h])
    pillar_8 = box(pillar_mid, [offset_mid_x, length/2-pillar_mid[1]/2, floor_h])
    pillar_9 = box(pillar_mid, [offset_mid_x, length-pillar_mid[1], floor_h])

    # connections in y
    connection = np.array([width, 145, 22])
    connection_1 = box(connection, [0, 0, floor_h+pillar[2]])
    connection_2 = box(connection, [0, length/2-connection[1]/2, floor_h+pillar[2]])
    connection_3 = box(connection, [0, length-connection[1], floor_h+pillar[2]])

    # top wide
    top_wide = np.array([145, length, 22])
    top_wide_1 = box(top_wide, [0, 0, floor_h+pillar[2]+connection[2]])
    top_wide_2 = box(top_wide, [width/2-top_wide[0]/2, 0, floor_h+pillar[2]+connection[2]])
    top_wide_3 = box(top_wide, [width-top_wide[0], 0, floor_h+pillar[2]+connection[2]])

    # top narrow
    top_narrow = np.array([100, length, 22])
    center_narrow = (offset_mid_x-top_wide[0])/2-top_narrow[0]/2
    top_narrow_1 = box(top_narrow, [top_wide[0]+center_narrow, 0, floor_h+pillar[2]+connection[2]])
    top_narrow_2 = box(top_narrow, [offset_mid_x+top_wide[0]+center_narrow, 0, floor_h+pillar[2]+connection[2]])

    objects = [
        floor1, floor2, floor3,
        pillar_1, pillar_2, pillar_3, pillar_4,
        pillar_5, pillar_6,
        pillar_7, pillar_8, pillar_9,
        connection_1, connection_2, connection_3,
        top_wide_1, top_wide_2, top_wide_3,
        top_narrow_1, top_narrow_2
    ]

    palette = trimesh.util.concatenate(objects)
    # reference point to upper end
    palette = palette.apply_translation([0, 0, -floor_outer[2]-pillar[2]-connection[2]-top_wide[2]])
    # scale to given scaling
    width_factor, length_factor = width_scaling / width, length_scaling / length
    scale_width = trimesh.transformations.scale_matrix(width_factor, direction=[1, 0, 0])
    scale_length = trimesh.transformations.scale_matrix(length_scaling / length, direction=[0, 1, 0])
    scale_height = trimesh.transformations.scale_matrix(max(width_factor, length_factor), direction=[0, 0, 1])
    palette = palette.apply_transform(scale_width).apply_transform(scale_length).apply_transform(scale_height)
    # scale from mm to cm
    palette = palette.apply_scale(0.1)
    return palette
