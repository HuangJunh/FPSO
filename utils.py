import configparser
import logging
import time
import sys
import os
from subprocess import Popen, PIPE

class Utils(object):
    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_params('settings', 'pop_size')
        params['num_iteration'] = cls.get_params('settings', 'num_iteration')
        params['init_max_length'] = cls.get_params('network', 'init_max_length')
        params['mean_length'] = cls.get_params('network', 'mean_length')
        params['stddev_length'] = cls.get_params('network', 'stddev_length')
        params['max_pool'] = cls.get_params('network', 'max_pool')
        params['image_channel'] = cls.get_params('network', 'image_channel')
        params['max_output_channel'] = cls.get_params('network', 'max_output_channel')
        params['num_class'] = cls.get_params('network', 'num_class')
        params['min_epoch_eval'] = cls.get_params('network', 'min_epoch_eval')
        params['epoch_test'] = cls.get_params('network', 'epoch_test')
        return params

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def get_params(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return int(rs)

    @classmethod
    def calc_parameters_num(cls, particle):
        num = 0
        for i,dimen in enumerate(particle):
            if 0 <= dimen <= 127:
                if i == 0:
                    in_channel = cls.get_params('network', 'image_channel')
                else:
                    in_channel = pre_out_channel
                num += 3 * 3 * in_channel * (dimen+1) + 2 * (dimen+1)
                pre_out_channel = dimen + 1
        num += pre_out_channel*cls.get_params('network', 'num_class')
        return num

    @classmethod
    def save_population(cls, type, population, gen_no):
        file_name = './populations/'+type+'_%02d.txt' % (gen_no)
        _str = cls.pop2str(population)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_acc(cls, type, acc_set, gen_no):
        file_name = './populations/'+type+'_acc_%02d.txt' % (gen_no)
        _str = cls.acc2str(acc_set)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_and_acc(cls, type, population, acc_set, gen_no):
        file_name = './populations/' + type + '_%02d.txt' % (gen_no)
        _str = cls.popAndAcc2str(population, acc_set)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()

    @classmethod
    def pop2str(cls, population):
        pop_str = []
        for id,particle in enumerate(population):
            _str = []
            _str.append('indi:%02d' % (id))
            _str.append('particle:%s' % ('.'.join(list(map(str, particle)))))
            for number, dimen in enumerate(particle):
                _sub_str = []
                if 0<= dimen <=127:
                    _sub_str.append('conv')
                    _sub_str.append('number:%d' % (number))
                    if number == 0:
                        in_channel = cls.get_params('network', 'image_channel')
                    else:
                        in_channel = pre_out_channel
                    _sub_str.append('in:%d' % (in_channel))
                    _sub_str.append('out:%d' % (dimen+1))
                    pre_out_channel = dimen+1

                if 256<= dimen <=383:
                    _sub_str.append('pool')
                    _sub_str.append('number:%d' % (number))
                    if 256<= dimen <= 319:
                        _sub_str.append('type:%s' % ('max'))
                    else:
                        _sub_str.append('type:%s' % ('average'))

                _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
            particle_str = '\n'.join(_str)
            pop_str.append(particle_str)
            pop_str.append('-' * 100)
        return '\n'.join(pop_str)


    @classmethod
    def acc2str(cls, acc_set):
        acc_str = []
        for id,acc in enumerate(acc_set):
            _sub_str = []
            _sub_str.append('indi:%02d' % (id))
            _sub_str.append('acc:%s' % (acc))
            acc_str.append('%s'%(','.join(_sub_str)))
            acc_str.append('-' * 100)
        return '\n'.join(acc_str)

    @classmethod
    def popAndAcc2str(cls, population, acc_set):
        pop_str = []
        for id, particle in enumerate(population):
            _str = []
            _str.append('indi:%02d' % (id))
            _str.append('particle:%s' % ('.'.join(list(map(str, particle)))))
            _str.append('num_parameters:%d' % (cls.calc_parameters_num(particle)))
            _str.append('eval_acc:%.4f' % (acc_set[id]))
            for number, dimen in enumerate(particle):
                _sub_str = []
                if 0 <= dimen <= 127:
                    _sub_str.append('conv')
                    _sub_str.append('number:%d' % (number))
                    if number == 0:
                        in_channel = cls.get_params('network', 'image_channel')
                    else:
                        in_channel = pre_out_channel
                    _sub_str.append('in:%d' % (in_channel))
                    _sub_str.append('out:%d' % (dimen+1))
                    pre_out_channel = dimen+1

                if 256 <= dimen <= 383:
                    _sub_str.append('pool')
                    _sub_str.append('number:%d' % (number))
                    if 256 <= dimen <= 319:
                        _sub_str.append('type:%s' % ('max'))
                    else:
                        _sub_str.append('type:%s' % ('average'))

                _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
            particle_str = '\n'.join(_str)
            pop_str.append(particle_str)
            pop_str.append('-' * 100)
        return '\n'.join(pop_str)

    @classmethod
    def read_template(cls):
        dataset = str(cls.__read_ini_file('settings', 'dataset'))
        _path = './template/' + dataset + '.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, particle, curr_gen, id):
        # query convolution unit
        conv_name_list = []
        conv_list = []
        in_channel_list = []
        out_channel_list = []
        for i, dimen in enumerate(particle):
            if 0 <= dimen <= 127:
                conv_name = 'self.conv_%d' % (i)

                if i == 0:
                    in_channel = cls.get_params('network', 'image_channel')
                else:
                    in_channel = pre_out_channel

                if conv_name not in conv_name_list:
                    conv_name_list.append(conv_name)
                    conv = '%s = BasicBlock(in_planes=%d, planes=%d)' % (conv_name, in_channel, dimen+1)
                    conv_list.append(conv)
                pre_out_channel = dimen+1
                in_channel_list.append(in_channel)
                out_channel_list.append(dimen+1)
            else:
                in_channel_list.append(out_channel_list[-1])
                out_channel_list.append(out_channel_list[-1])

        fully_layer_name = 'self.linear = nn.Linear(%d, %d)' % (out_channel_list[-1], cls.get_params('network', 'num_class'))

        # generate the forward part
        forward_list = []
        for i, dimen in enumerate(particle):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d' % (i - 1)
            if 0 <= dimen <= 127:
                _str = 'out_%d = self.conv_%d(%s)' % (i, i, last_out_put)
                forward_list.append(_str)

            else:
                if 256 <= dimen <= 319:
                    _str = 'out_%d = F.max_pool2d(out_%d, 2)' % (i, i - 1)
                else:
                    _str = 'out_%d = F.avg_pool2d(out_%d, 2)' % (i, i - 1)
                forward_list.append(_str)
        forward_list.append('out = out_%d' % (len(particle) - 1))
        forward_list.append('out = F.adaptive_avg_pool2d(out,(1,1))')

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
        file_path = './scripts/particle%02d_%02d.py' % (curr_gen, id)
        script_file_handler = open(file_path, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        file_name = 'particle%02d_%02d'%(curr_gen, id)
        return file_name


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("FPSO")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class GPUTools(object):
    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_info(cls):
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        equipped_gpu_ids = []
        for line_info in lines:
            if not line_info.startswith(' '):
                if 'GeForce' in line_info:
                    equipped_gpu_ids.append(line_info.strip().split(' ', 4)[3])
            else:
                break

        gpu_info_list = []
        for line_no in range(len(lines) - 3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())

        return equipped_gpu_ids, gpu_info_list

    @classmethod
    def get_available_gpu_ids(cls):
        equipped_gpu_ids, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))

        unused_gpu_ids = []
        for id_ in equipped_gpu_ids:
            if id_ not in used_gpu_ids:
                unused_gpu_ids.append(id_)
        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        unused_gpu_ids = cls.get_available_gpu_ids()
        if len(unused_gpu_ids) == 0:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            Log.info('GPU_QUERY-Available GPUs are: [%s], choose GPU#%s to use' % (','.join(unused_gpu_ids), unused_gpu_ids[0]))
            return int(unused_gpu_ids[0])

    @classmethod
    def all_gpu_available(cls):
        _, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))
        if len(used_gpu_ids) == 0:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            Log.info('GPU_QUERY- GPUs [%s] are occupying' % (','.join(used_gpu_ids)))
            return False


