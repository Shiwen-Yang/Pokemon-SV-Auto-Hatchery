import nxbt

def nx_init(address = 'DC:68:EB:76:B0:35'):
    # Init NXBT
    nx = nxbt.Nxbt()
    # print(nx.get_switch_addresses())
    adapter = nx.get_available_adapters()[0]
    bt_address = address
    ctrler = nx.create_controller(
            nxbt.PRO_CONTROLLER,
            adapter_path = adapter,
            colour_body = [0,0,0],
            colour_buttons = [255,255,255], 
            reconnect_address = bt_address
            )
    nx.wait_for_connection(ctrler)
    return(nx, ctrler)