function f_cs_startup(app)
disp('Caiman Sorter GUI started');

app.VisualizationparamsPanel.Visible = 0;
app.CatracePanel.Visible = 0;
app.TabGroup.Visible = 0;
app.TabGroup2.Visible = 0;
app.CellselectionPanel.Visible = 0;

ops = f_cs_collect_ops(app);
ops_path = app.ops_path;
ops.ops_path = ops_path;

if exist(ops_path, 'file')
    ops_temp = load(ops_path, 'ops');
    ops = ops_temp.ops;
    ops.ops_path = ops_path;
    f_cs_update_log(app, ['ops loaded path: ' ops.ops_path]);
    if ~isfield(ops_temp.ops, 'ops_path') || ~strcmpi(ops_temp.ops.ops_path, ops_path)
        disp('RAFA: Welcome new user, you should join the Yuste lab, it is great, no?');
        f_cs_update_log(app, 'RAFA: Welcome new user, you should join the Yuste lab, it is great, no?');
    end
    f_cs_write_ops(app);
else
    save(ops_path, 'ops');
    disp('RAFA: Welcome new user, you should join the Yuste lab, it is great, no?');
    f_cs_update_log(app, 'RAFA: Welcome new user, you should join the Yuste lab, it is great, no?');
end
if ops.file_path_from_python~=0
    app.file_loc=ops.file_path_from_python ;
    app.ops.browse_path=fileparts(app.file_loc)
    f_cs_load_button_pushed(app);
end

app.ops = ops;

end