#!/usr/bin/env python3
# -*-coding:utf8-*-
# can总线读取二次封装
# 反馈码为100开头，反馈码总长为000000
import can
from can.message import Message
from can.interfaces.socketcand.socketcand import (
    SocketCanDaemonBus,
    convert_ascii_message_to_can_message,
)
import platform
import socket
import time
from threading import Timer
import subprocess
from typing import (
    Callable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from enum import IntEnum, auto


class _SocketCanDaemonBus(SocketCanDaemonBus):
    """socketcand client whose handshake handles TCP message coalescing.

    python-can 4.6.1 expects one ``recv`` call to contain exactly ``< ok >``.
    A busy socketcand can legally return ``< ok >< frame ... >`` in one TCP
    segment.  Preserve and queue the trailing CAN frames instead of rejecting
    the connection.
    """

    def _expect_msg(self, expected):
        receive_buffer_name = "_SocketCanDaemonBus__receive_buffer"
        socket_name = "_SocketCanDaemonBus__socket"
        tcp_tune_name = "_SocketCanDaemonBus__tcp_tune"
        message_buffer_name = "_SocketCanDaemonBus__message_buffer"

        receive_buffer = getattr(self, receive_buffer_name)
        connection = getattr(self, socket_name)
        while ">" not in receive_buffer:
            chunk = connection.recv(256)
            if not chunk:
                raise can.CanError(
                    f"socketcand closed while waiting for '{expected}'"
                )
            receive_buffer += chunk.decode("ascii")
            if getattr(self, tcp_tune_name):
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)

        response_end = receive_buffer.index(">") + 1
        response = receive_buffer[:response_end]
        trailing = receive_buffer[response_end:]
        if response != expected:
            raise can.CanError(f"Expected '{expected}' got: '{response}'")

        queued_messages = getattr(self, message_buffer_name)
        while trailing:
            frame_start = trailing.find("<")
            if frame_start == -1:
                trailing = ""
                break
            frame_end = trailing.find(">", frame_start)
            if frame_end == -1:
                trailing = trailing[frame_start:]
                break
            frame_text = trailing[frame_start : frame_end + 1]
            frame = convert_ascii_message_to_can_message(frame_text)
            if frame is not None:
                frame.channel = self.channel
                queued_messages.append(frame)
            trailing = trailing[frame_end + 1 :]
        setattr(self, receive_buffer_name, trailing)

class C_STD_CAN():
    '''
    基础CAN数据帧的收发,内无线程创建,需要在类外调用的时候创建线程来循环read
    
    Args:
        channel_name: can的端口名称
        bustype: can总线类型,默认为socket can
        expected_bitrate: 预期can总线的波特率
        judge_flag: 是否在实例化该类时进行can端口判断,有些情况需要False 
        auto_init: 是否自动初始化can,也就是实例化can.interface.Bus
        callback_function: ReadCanMessage中的回调函数,应传入函数
    '''
    '''
    Basic CAN Frame Send/Receive with Thread Creation

    When calling outside the class, a thread needs to be created to continuously read the CAN data.

    Args:
        channel_name: The name of the CAN port.
        bustype: The type of CAN bus, default is socket CAN.
        expected_bitrate: The expected bitrate for the CAN bus.
        judge_flag: Whether to check the CAN port during the instantiation of the class. In some cases, it should be set to False.
        auto_init: Whether to automatically initialize the CAN bus (i.e., instantiate can.interface.Bus).
        callback_function: The callback function in ReadCanMessage, which should be passed as a function.
    '''
    class CAN_STATUS(IntEnum):
        # __del__
        DEL_CAN_BUS_CONNECT_SHUT_DOWN = 100001
        DEL_CAN_BUS_WAS_NOT_PROPERLY_INIT = auto()
        DEL_SHUTTING_DOWN_CAN_BUS_ERR = auto()
        INIT_CAN_BUS_IS_EXIST = auto()
        INIT_CAN_BUS_OPENED_SUCCESS = auto()
        INIT_CAN_BUS_OPENED_FAILED = auto()
        CLOSE_CAN_BUS_CONNECT_SHUT_DOWN = auto()
        CLOSE_CAN_BUS_WAS_NOT_PROPERLY_INIT = auto()
        CLOSE_SHUTTING_DOWN_CAN_BUS_ERR = auto()
        CLOSED_CAN_BUS_NOT_OPEN = auto()
        JUDGE_PASS = auto()
        READ_CAN_MSG_OK = auto()
        READ_CAN_MSG_TIMEOUT = auto()
        READ_CAN_MSG_FAILED = auto()
        READ_CAN_BUS_NOT_OK = auto()
        SEND_MESSAGE_SUCCESS = auto()
        SEND_MESSAGE_FAILED = auto()
        SEND_CAN_BUS_NOT_OK = auto()
        BUS_STATE_ACTIVE = auto()
        BUS_STATE_PASSIVE = auto()
        BUS_STATE_ERROR = auto()
        BUS_STATE_UNKNOWN = auto()
        CHECK_CAN_EXIST = auto()
        CHECK_CAN_UP = auto()
        CHECK_CAN_NOT_UP = auto()
        CAN_SOCKET_NOT_EXIST = auto()
        CAN_BITRATE_SUCCESS= auto()
        CAN_BITRATE_ERR= auto()
        def __str__(self):
            return f"{self.name} ({self.value})"
        def __repr__(self):
            return f"{self.name}: {self.value}"
    
    def __init__(self, 
                 channel_name:str="can0", 
                 bustype="socketcan", 
                 expected_bitrate:int=1000000,
                 judge_flag:bool=True, 
                 auto_init:bool=True,
                 callback_function: Callable = None,
                 frame_observer: Optional[Callable] = None,
                 **bus_kwargs) -> None:
        self.channel_name = channel_name
        self.bustype = bustype
        self.expected_bitrate = expected_bitrate
        self.bus_kwargs = dict(bus_kwargs)
        self.rx_message:Optional[Message] = Message()   #创建消息接收类
        self.callback_function = callback_function  #接收回调函数
        # observer(direction, message, local_start_ns, local_end_ns, result, error)
        # Local times are monotonic; message.timestamp belongs to the CAN source.
        self.frame_observer = frame_observer
        self.recv_bus = None
        self.send_bus = None
        self._share_bus_between_rx_tx = (
            self.bustype == "socketcand"
            or platform.system() in ("Windows", "Darwin")
        )
        if(judge_flag):
            self.JudgeCanInfo()
        if(auto_init):
            self.Init()#创建can总线交互
    
    def __del__(self):
        try:
            self._shutdown_buses()
            return self.CAN_STATUS.DEL_CAN_BUS_CONNECT_SHUT_DOWN
        except AttributeError:
            return self.CAN_STATUS.DEL_CAN_BUS_WAS_NOT_PROPERLY_INIT
        except Exception as e:
            return self.CAN_STATUS.DEL_SHUTTING_DOWN_CAN_BUS_ERR
    
    def Init(self):
        '''初始化can总线
        '''
        '''Initialize the CAN bus.
        '''
        if self.recv_bus is not None and self.send_bus is not None:
            # return True
            return self.CAN_STATUS.INIT_CAN_BUS_IS_EXIST
        try:
            self.recv_bus = self._create_bus()
            if self._share_bus_between_rx_tx:
                self.send_bus = self.recv_bus
            else:
                self.send_bus = self._create_bus()
            return self.CAN_STATUS.INIT_CAN_BUS_OPENED_SUCCESS
        except can.CanError as e:
            try:
                self._shutdown_buses()
            except Exception:
                pass
            self.recv_bus = None
            self.send_bus = None
            raise

    def Close(self):
        '''关闭can总线
        '''
        '''Close the CAN bus.
        '''
        if self.recv_bus is not None and self.send_bus is not None:
            try:
                self._shutdown_buses()
                self.recv_bus = None
                self.send_bus = None
                # return True
                return self.CAN_STATUS.CLOSE_CAN_BUS_CONNECT_SHUT_DOWN
            except AttributeError:
                return self.CAN_STATUS.CLOSE_CAN_BUS_WAS_NOT_PROPERLY_INIT
            except Exception as e:
                return self.CAN_STATUS.CLOSE_SHUTTING_DOWN_CAN_BUS_ERR
            # return 1
        else:
            return self.CAN_STATUS.CLOSED_CAN_BUS_NOT_OPEN

    def _create_bus(self):
        bus_kwargs = dict(self.bus_kwargs)
        if self.bustype == "socketcand":
            return _SocketCanDaemonBus(
                channel=self.channel_name,
                **bus_kwargs,
            )
        if self.bustype == "socketcan":
            bus_kwargs.setdefault("bitrate", self.expected_bitrate)
            bus_kwargs.setdefault("receive_own_messages", False)
            bus_kwargs.setdefault("local_loopback", False)
        return can.interface.Bus(
            channel=self.channel_name,
            interface=self.bustype,
            **bus_kwargs,
        )

    def _shutdown_buses(self):
        if self.recv_bus is None and self.send_bus is None:
            raise AttributeError("CAN bus was not initialized.")

        shutdown_bus_ids = set()
        for bus in (self.recv_bus, self.send_bus):
            if bus is None or id(bus) in shutdown_bus_ids:
                continue
            bus.shutdown()
            shutdown_bus_ids.add(id(bus))
    
    def JudgeCanInfo(self):
        '''
        类初始化时是否检测基础信息
        '''
        '''
        Whether to check basic information during class initialization.
        '''
        # 检查 CAN 端口是否存在
        if self.is_can_socket_available(self.channel_name) is not self.CAN_STATUS.CHECK_CAN_EXIST:
            raise ValueError(f"CAN socket {self.channel_name} does not exist.")
        # 检查 CAN 端口是否 UP
        if self.is_can_port_up(self.channel_name) is not self.CAN_STATUS.CHECK_CAN_UP:
            raise RuntimeError(f"CAN port {self.channel_name} is not UP.")
        # 检查 CAN 端口的比特率
        actual_bitrate = self.get_can_bitrate(self.channel_name)
        if self.expected_bitrate is not None and not (actual_bitrate == self.expected_bitrate):
            raise ValueError(f"CAN port {self.channel_name} bitrate is {actual_bitrate} bps, expected {self.expected_bitrate} bps.")
        # return True
        return self.CAN_STATUS.JUDGE_PASS
    
    def GetBirtrate(self):
        return self.expected_bitrate

    def GetRxMessage(self) -> Message:
        return self.rx_message
    
    def GetCanPortName(self):
        return self.channel_name

    def ReadCanMessage(self):
        can_bus_status = self.is_can_bus_ok(self.recv_bus)
        if(can_bus_status == self.CAN_STATUS.BUS_STATE_ACTIVE):
            try:
                self.rx_message = self.recv_bus.recv(1)
                received_ns = time.monotonic_ns()
            except Exception:
                return self.CAN_STATUS.READ_CAN_MSG_FAILED
            if self.rx_message is None:
                return self.CAN_STATUS.READ_CAN_MSG_TIMEOUT
            if self.frame_observer is not None:
                self.frame_observer(
                    "rx", self.rx_message, received_ns, received_ns, "received", ""
                )
            try:
                if self.rx_message and self.callback_function:
                    self.callback_function(self.rx_message) #回调函数处理接收的原始数据
                return self.CAN_STATUS.READ_CAN_MSG_OK
            except Exception as e:
                return self.CAN_STATUS.READ_CAN_MSG_FAILED
        else:
            return can_bus_status

    def SendCanMessage(self, arbitration_id, data, dlc=8, is_extended_id=False):
        '''can transmit

        Args:
            arbitration_id (_type_): _description_
            data (_type_): _description_ Defaults to 8.
            is_extended_id_ (bool, optional): _description_. Defaults to False.
        '''
        message = can.Message(channel=self.channel_name,
                              arbitration_id=arbitration_id, 
                              data=data, 
                              dlc=dlc,
                              is_extended_id=is_extended_id)
        started_ns = time.monotonic_ns()
        error = ""
        if(self.is_can_bus_ok(self.send_bus) == self.CAN_STATUS.BUS_STATE_ACTIVE):
            try:
                self.send_bus.send(message)
                result = self.CAN_STATUS.SEND_MESSAGE_SUCCESS
            except Exception as e:
                result = self.CAN_STATUS.SEND_MESSAGE_FAILED
                error = str(e)
        else:
            result = self.CAN_STATUS.SEND_CAN_BUS_NOT_OK
        finished_ns = time.monotonic_ns()
        if self.frame_observer is not None:
            self.frame_observer(
                "tx", message, started_ns, finished_ns, result.name, error
            )
        return result

    def is_can_bus_ok(self, bus=None) -> bool:
        '''
        检查CAN总线状态是否正常。
        '''
        '''
        Check whether the CAN bus status is normal.
        '''
        if isinstance(bus, can.BusABC):
            bus_state = bus.state
        else: bus_state = None
        if bus_state == can.BusState.ACTIVE:
            # return True
            return self.CAN_STATUS.BUS_STATE_ACTIVE
        elif bus_state == can.BusState.PASSIVE:
            # return False
            return self.CAN_STATUS.BUS_STATE_PASSIVE
        elif bus_state == can.BusState.ERROR:
            # return False
            return self.CAN_STATUS.BUS_STATE_ERROR
        else:
            # return False
            return self.CAN_STATUS.BUS_STATE_UNKNOWN
    
    def is_can_socket_available(self, channel_name: str) -> bool:
        '''
        检查给定的 CAN 端口是否存在。
        '''
        '''
        Check if the given CAN port exists.
        '''
        try:
            with open(f"/sys/class/net/{channel_name}/operstate", "r") as file:
                state = file.read().strip()
                # return True
                return  self.CAN_STATUS.CHECK_CAN_EXIST
        except FileNotFoundError:
            # return False
            return  self.CAN_STATUS.CAN_SOCKET_NOT_EXIST
    
    def is_can_port_up(self, channel_name: str) -> bool:
        '''
        检查 CAN 端口是否为 UP 状态。
        '''
        '''
        Check if the CAN port is in the UP state.
        '''
        try:
            with open(f"/sys/class/net/{channel_name}/operstate", "r") as file:
                state = file.read().strip()
                if(state == "up"):
                    # return True
                    return  self.CAN_STATUS.CHECK_CAN_UP
                else: 
                    # return False
                    return  self.CAN_STATUS.CHECK_CAN_NOT_UP
        except FileNotFoundError:
            # return False
            return  self.CAN_STATUS.CAN_SOCKET_NOT_EXIST

    def get_can_ports(self) -> list:
        '''
        获取系统中所有可用的 CAN 端口。
        '''
        '''
        Get all available CAN ports in the system.
        '''
        import os
        can_ports = []
        for item in os.listdir('/sys/class/net/'):
            if 'can' in item:
                can_ports.append(item)
        return can_ports

    def can_port_info(self, channel_name: str) -> str:
        '''
        获取指定 CAN 端口的详细信息，包括状态、类型和比特率。
        '''
        '''
        Get detailed information about the specified CAN port, including status, type, and bit rate.
        '''
        try:
            with open(f"/sys/class/net/{channel_name}/operstate", "r") as file:
                state = file.read().strip()
            with open(f"/sys/class/net/{channel_name}/type", "r") as file:
                port_type = file.read().strip()
            bitrate = self.get_can_bitrate(channel_name)
            return f"CAN port {channel_name}: State={state}, Type={port_type}, Bitrate={bitrate} bps"
        except FileNotFoundError:
            return f"CAN port {channel_name} not found."

    def get_can_bitrate(self, channel_name: str) -> str:
        '''
        获取指定 CAN 端口的比特率。
        '''
        '''
        Get the bit rate of the specified CAN port.
        '''
        try:
            result = subprocess.run(['ip', '-details', 'link', 'show', channel_name],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, check=True)  # Python 3.6
                                    # capture_output=True, text=True)
            output = result.stdout
            for line in output.split('\n'):
                if 'bitrate' in line:
                    return int(line.split('bitrate ')[1].split(' ')[0])
            return self.CAN_STATUS.CAN_BITRATE_SUCCESS
        except Exception as e:
            return self.CAN_STATUS.CAN_BITRATE_ERR, e

## 示例代码
# if __name__ == "__main__":
#     can_name = "can0"
#     try:
#         can_obj = C_STD_CAN(channel_name=can_name)
#         print("CAN bus initialized successfully.")
#         print(can_obj.get_can_ports())
#         print(can_obj.can_port_info(can_name))
#         print(can_obj.ReadCanMessage())
#         print(can_obj.GetRxMessage())
#         # print(can_obj.get_can_bitrate("can_name"))
#         print(f"{can_obj.CAN_STATUS.SEND_CAN_BUS_NOT_OK}")
#     except ValueError as e:
#         print(e)
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     can_name = "vcan0"
#     bus = C_STD_CAN(can_name, "socketcan", 1000000,False, True)
#     while True:
#         start_time = time.time()
#         # piper.SearchPiperFirmwareVersion()
#         bus.SendCanMessage(0x101, [0,0,0,0,0,0,0,0])
#         end_time = time.time()
#         cost_ms = (end_time - start_time) * 1000  # 转为毫秒
#         if(cost_ms > 1):
#             print(f"[MAX UPDATE] 最大执行耗时: {cost_ms:.3f} ms")
#         time.sleep(0.001)
#     bus.shutdown()
